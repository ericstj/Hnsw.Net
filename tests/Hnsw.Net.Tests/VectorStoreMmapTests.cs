using Microsoft.Extensions.VectorData;
using HnswNet;
using Xunit;

namespace Hnsw.Net.Tests;

public partial class VectorStoreTests
{
    [Fact]
    public async Task LoadMapped_FromPath_RoundTripsRecordsAndSearch()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();
        string path = Path.GetTempFileName();
        try
        {
            using (FileStream fs = File.Create(path))
            {
                collection.Save(fs, SnapshotContext.Default);
            }

            using var store = new HnswVectorStore();
            HnswCollection<int, Doc> reloaded = store.GetCollection<int, Doc>("docs");
            reloaded.Load(path, SnapshotContext.Default);

            // Lazily materialized record payload.
            Doc? fetched = await reloaded.GetAsync(2);
            Assert.NotNull(fetched);
            Assert.Equal("y axis", fetched!.Text);

            var results = new List<VectorSearchResult<Doc>>();
            await foreach (VectorSearchResult<Doc> r in reloaded.SearchAsync(new float[] { 0.9f, 0.1f, 0f }, top: 3))
            {
                results.Add(r);
            }

            Assert.Equal(3, results.Count);
            Assert.Equal(1, results[0].Record.Id);
        }
        finally
        {
            // store is disposed by `using` above, releasing the mapping so the file can be deleted (Windows locks maps).
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public async Task LoadMapped_ByReflection_RoundTrips()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();
        string path = Path.GetTempFileName();
        try
        {
            using (FileStream fs = File.Create(path))
            {
                collection.Save(fs, SnapshotContext.Default);
            }

            using var store = new HnswVectorStore();
            HnswCollection<int, Doc> reloaded = store.GetCollection<int, Doc>("docs");
            reloaded.Load(path);

            Doc? fetched = await reloaded.GetAsync(3);
            Assert.NotNull(fetched);
            Assert.Equal("z axis", fetched!.Text);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task LoadMapped_AtOffset_RoundTrips()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();
        string path = Path.GetTempFileName();
        const int prefix = 13;
        try
        {
            using (FileStream fs = File.Create(path))
            {
                fs.Write(new byte[prefix]);
                collection.Save(fs, SnapshotContext.Default);
            }

            using var store = new HnswVectorStore();
            HnswCollection<int, Doc> reloaded = store.GetCollection<int, Doc>("docs");
            reloaded.Load(path, prefix, SnapshotContext.Default);

            Doc? fetched = await reloaded.GetAsync(1);
            Assert.NotNull(fetched);
            Assert.Equal("x axis", fetched!.Text);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task LoadMapped_MatchesStreamLoadResults()
    {
        HnswCollection<int, Doc> collection = await SeedAsync();
        string path = Path.GetTempFileName();
        try
        {
            using (FileStream fs = File.Create(path))
            {
                collection.Save(fs, SnapshotContext.Default);
            }

            using var streamStore = new HnswVectorStore();
            HnswCollection<int, Doc> streamLoaded = streamStore.GetCollection<int, Doc>("docs");
            using (FileStream fs = File.OpenRead(path))
            {
                streamLoaded.Load(fs, SnapshotContext.Default);
            }

            using var mapStore = new HnswVectorStore();
            HnswCollection<int, Doc> mapLoaded = mapStore.GetCollection<int, Doc>("docs");
            mapLoaded.Load(path, SnapshotContext.Default);

            float[] query = { 0f, 0.2f, 0.9f };
            Assert.Equal(await CollectIdsAsync(streamLoaded, query), await CollectIdsAsync(mapLoaded, query));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static async Task<List<int>> CollectIdsAsync(HnswCollection<int, Doc> collection, float[] query)
    {
        var ids = new List<int>();
        await foreach (VectorSearchResult<Doc> r in collection.SearchAsync(query, top: 3))
        {
            ids.Add(r.Record.Id);
        }

        return ids;
    }
}
