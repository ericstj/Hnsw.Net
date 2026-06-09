using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using System.Buffers.Binary;
using System.Text.Json.Serialization;
using HnswNet;
using Xunit;

namespace Hnsw.Net.Tests;

public partial class VectorStoreTests
{
    private sealed class Doc
    {
        [VectorStoreKey]
        public int Id { get; set; }

        [VectorStoreData]
        public string Text { get; set; } = string.Empty;

        [VectorStoreData]
        public string Category { get; set; } = string.Empty;

        [VectorStoreVector(3, DistanceFunction = DistanceFunction.CosineSimilarity)]
        public ReadOnlyMemory<float> Vector { get; set; }
    }

    private static async Task<HnswCollection<int, Doc>> SeedAsync()
    {
        var store = new HnswVectorStore();
        HnswCollection<int, Doc> collection = store.GetCollection<int, Doc>("docs");
        await collection.EnsureCollectionExistsAsync();

        await collection.UpsertAsync(new[]
        {
            new Doc { Id = 1, Text = "x axis", Category = "axis", Vector = new float[] { 1f, 0f, 0f } },
            new Doc { Id = 2, Text = "y axis", Category = "axis", Vector = new float[] { 0f, 1f, 0f } },
            new Doc { Id = 3, Text = "z axis", Category = "other", Vector = new float[] { 0f, 0f, 1f } },
        });

        return collection;
    }

    [Fact]
    public async Task Upsert_Get_RoundTrips()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        Doc? fetched = await collection.GetAsync(2);

        Assert.NotNull(fetched);
        Assert.Equal("y axis", fetched!.Text);
    }

    [Fact]
    public async Task Search_ReturnsNearestFirst()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        List<VectorSearchResult<Doc>> results = new();
        await foreach (VectorSearchResult<Doc> r in collection.SearchAsync(new float[] { 0.9f, 0.1f, 0f }, top: 3))
        {
            results.Add(r);
        }

        Assert.Equal(3, results.Count);
        Assert.Equal(1, results[0].Record.Id);
        Assert.True(results[0].Score > results[1].Score);
    }

    [Fact]
    public async Task Search_RespectsFilter()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        List<VectorSearchResult<Doc>> results = new();
        var options = new VectorSearchOptions<Doc> { Filter = d => d.Category == "other" };
        await foreach (VectorSearchResult<Doc> r in collection.SearchAsync(new float[] { 1f, 0f, 0f }, top: 3, options))
        {
            results.Add(r);
        }

        Assert.Single(results);
        Assert.Equal(3, results[0].Record.Id);
    }

    [Fact]
    public async Task Delete_RemovesRecord()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        await collection.DeleteAsync(2);

        Assert.Null(await collection.GetAsync(2));

        List<VectorSearchResult<Doc>> results = new();
        await foreach (VectorSearchResult<Doc> r in collection.SearchAsync(new float[] { 0f, 1f, 0f }, top: 3))
        {
            results.Add(r);
        }

        Assert.DoesNotContain(results, r => r.Record.Id == 2);
    }

    [Fact]
    public async Task Upsert_Update_ReplacesVector()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        await collection.UpsertAsync(new Doc { Id = 1, Text = "moved", Category = "axis", Vector = new float[] { 0f, 0f, 1f } });

        List<VectorSearchResult<Doc>> results = new();
        await foreach (VectorSearchResult<Doc> r in collection.SearchAsync(new float[] { 0f, 0f, 1f }, top: 2))
        {
            results.Add(r);
        }

        Assert.Equal("moved", (await collection.GetAsync(1))!.Text);
        Assert.Contains(results, r => r.Record.Id == 1);
    }

    [Fact]
    public async Task SaveLoad_RoundTripsRecordsAndSearch()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        using var ms = new MemoryStream();
        collection.Save(ms);
        ms.Position = 0;

        var reloaded = new HnswVectorStore().GetCollection<int, Doc>("docs");
        reloaded.Load(ms);

        Doc? fetched = await reloaded.GetAsync(2);
        Assert.NotNull(fetched);
        Assert.Equal("y axis", fetched!.Text);

        List<VectorSearchResult<Doc>> results = new();
        await foreach (VectorSearchResult<Doc> r in reloaded.SearchAsync(new float[] { 0.9f, 0.1f, 0f }, top: 3))
        {
            results.Add(r);
        }

        Assert.Equal(3, results.Count);
        Assert.Equal(1, results[0].Record.Id);
    }

    [JsonSerializable(typeof(Doc))]
    [JsonSerializable(typeof(int))]
    private partial class SnapshotContext : JsonSerializerContext { }

    [Fact]
    public async Task SaveLoad_WithJsonContext_RoundTripsRecordsAndSearch()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        using var ms = new MemoryStream();
        collection.Save(ms, SnapshotContext.Default);
        ms.Position = 0;

        var reloaded = new HnswVectorStore().GetCollection<int, Doc>("docs");
        reloaded.Load(ms, SnapshotContext.Default);

        Doc? fetched = await reloaded.GetAsync(2);
        Assert.NotNull(fetched);
        Assert.Equal("y axis", fetched!.Text);

        List<VectorSearchResult<Doc>> results = new();
        await foreach (VectorSearchResult<Doc> r in reloaded.SearchAsync(new float[] { 0.9f, 0.1f, 0f }, top: 3))
        {
            results.Add(r);
        }

        Assert.Equal(3, results.Count);
        Assert.Equal(1, results[0].Record.Id);
    }

    [Fact]
    public async Task Save_WithJsonContext_LoadableByReflection()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();

        using var ms = new MemoryStream();
        collection.Save(ms, SnapshotContext.Default);
        ms.Position = 0;

        var reloaded = new HnswVectorStore().GetCollection<int, Doc>("docs");
        reloaded.Load(ms);

        Doc? fetched = await reloaded.GetAsync(3);
        Assert.NotNull(fetched);
        Assert.Equal("z axis", fetched!.Text);
    }

    [Fact]
    public async Task Load_DimensionMismatch_Throws()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();
        using var ms = new MemoryStream();
        collection.Save(ms);
        ms.Position = 0;

        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            {
                new VectorStoreKeyProperty(nameof(Doc.Id), typeof(int)),
                new VectorStoreVectorProperty(nameof(Doc.Vector), typeof(ReadOnlyMemory<float>), 5)
                {
                    DistanceFunction = DistanceFunction.CosineSimilarity,
                },
            },
        };
        var mismatched = new HnswVectorStore().GetCollection<int, Doc>("docs", definition);

        Assert.Throws<InvalidDataException>(() => mismatched.Load(ms));
    }

    [Fact]
    public async Task Load_MetricMismatch_Throws()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();
        using var ms = new MemoryStream();
        collection.Save(ms);
        ms.Position = 0;

        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            {
                new VectorStoreKeyProperty(nameof(Doc.Id), typeof(int)),
                new VectorStoreVectorProperty(nameof(Doc.Vector), typeof(ReadOnlyMemory<float>), 3)
                {
                    DistanceFunction = DistanceFunction.EuclideanDistance,
                },
            },
        };
        var mismatched = new HnswVectorStore().GetCollection<int, Doc>("docs", definition);

        Assert.Throws<InvalidDataException>(() => mismatched.Load(ms));
    }

    [Theory]
    [InlineData(8, -1)]   // dimension field
    public async Task Load_NegativeHeaderInt32_Throws(int offset, int value)
    {
        byte[] bytes = await SaveSnapshotAsync();
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset), value);

        var fresh = new HnswVectorStore().GetCollection<int, Doc>("docs");
        Assert.Throws<InvalidDataException>(() => fresh.Load(new MemoryStream(bytes)));
    }

    [Fact]
    public async Task Load_NegativeNextId_Throws()
    {
        byte[] bytes = await SaveSnapshotAsync();
        BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(16), -1L); // nextId field

        var fresh = new HnswVectorStore().GetCollection<int, Doc>("docs");
        Assert.Throws<InvalidDataException>(() => fresh.Load(new MemoryStream(bytes)));
    }

    [Fact]
    public async Task Load_NextIdNotGreaterThanRecordIds_Throws()
    {
        byte[] bytes = await SaveSnapshotAsync();
        BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(16), 0L); // nextId <= existing record ids

        var fresh = new HnswVectorStore().GetCollection<int, Doc>("docs");
        Assert.Throws<InvalidDataException>(() => fresh.Load(new MemoryStream(bytes)));
    }

    [Fact]
    public async Task Load_NonEmptyWithoutIndex_Throws()
    {
        byte[] bytes = await SaveSnapshotAsync();
        bytes[HasIndexOffset(bytes)] = 0; // clear hasIndex on a non-empty snapshot

        var fresh = new HnswVectorStore().GetCollection<int, Doc>("docs");
        Assert.Throws<InvalidDataException>(() => fresh.Load(new MemoryStream(bytes)));
    }

    [Fact]
    public async Task Load_DuplicateRecordId_Throws()
    {
        byte[] bytes = await SaveSnapshotAsync();
        long firstId = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(28));
        BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(SecondRecordOffset(bytes)), firstId);

        var fresh = new HnswVectorStore().GetCollection<int, Doc>("docs");
        Assert.Throws<InvalidDataException>(() => fresh.Load(new MemoryStream(bytes)));
    }

    // Walks the fixed 28-byte header and per-record framing to locate the trailing hasIndex byte.
    private static int HasIndexOffset(byte[] bytes)
    {
        int pos = 24;
        int count = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(pos));
        pos += 4;
        for (int i = 0; i < count; i++)
        {
            pos += 8; // id
            int keyLen = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(pos));
            pos += 4 + keyLen;
            int recLen = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(pos));
            pos += 4 + recLen;
        }

        return pos;
    }

    // Offset of the second record's id field (records start at byte 28).
    private static int SecondRecordOffset(byte[] bytes)
    {
        int pos = 28;
        pos += 8; // first record id
        int keyLen = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(pos));
        pos += 4 + keyLen;
        int recLen = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(pos));
        pos += 4 + recLen;
        return pos;
    }

    private static async Task<byte[]> SaveSnapshotAsync()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();
        using var ms = new MemoryStream();
        collection.Save(ms);
        return ms.ToArray();
    }

    private sealed class StringDoc
    {
        [VectorStoreKey]
        public int Id { get; set; }

        [VectorStoreVector(4, DistanceFunction = DistanceFunction.CosineSimilarity)]
        public string Text { get; set; } = string.Empty;
    }

    // Hashing generator: deterministic vectors per string, sufficient to verify the embedding-generation path.
    private sealed class HashingGenerator : IEmbeddingGenerator<string, Embedding<float>>
    {
        public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
            IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
        {
            var result = new GeneratedEmbeddings<Embedding<float>>();
            foreach (string value in values)
            {
                var vec = new float[4];
                foreach (char c in value)
                {
                    vec[c % 4] += 1f;
                }

                result.Add(new Embedding<float>(vec));
            }

            return Task.FromResult(result);
        }

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }

    [Fact]
    public async Task EmbeddingGeneration_ComposesWithGenerator()
    {
        var store = new HnswVectorStore(new HnswVectorStoreOptions { EmbeddingGenerator = new HashingGenerator() });
        HnswCollection<int, StringDoc> collection = store.GetCollection<int, StringDoc>("strings");
        await collection.EnsureCollectionExistsAsync();

        await collection.UpsertAsync(new[]
        {
            new StringDoc { Id = 1, Text = "aaaa" },
            new StringDoc { Id = 2, Text = "bbbb" },
            new StringDoc { Id = 3, Text = "cccc" },
        });

        List<VectorSearchResult<StringDoc>> results = new();
        await foreach (VectorSearchResult<StringDoc> r in collection.SearchAsync("aaaa", top: 1))
        {
            results.Add(r);
        }

        Assert.Single(results);
        Assert.Equal(1, results[0].Record.Id);
    }
}
