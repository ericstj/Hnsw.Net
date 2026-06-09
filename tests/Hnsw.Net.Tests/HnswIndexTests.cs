using System.Numerics.Tensors;
using HnswNet;
using Xunit;

namespace Hnsw.Net.Tests;

public sealed class HnswIndexTests
{
    [Theory]
    [InlineData(DistanceMetric.Cosine)]
    [InlineData(DistanceMetric.EuclideanL2)]
    [InlineData(DistanceMetric.DotProduct)]
    public void RecallMatchesBruteForce(DistanceMetric metric)
    {
        const int count = 2_000;
        const int dimension = 64;
        const int queryCount = 50;
        const int k = 10;
        float[][] vectors = RandomVectors(count, dimension, seed: 1234);
        float[][] queries = RandomVectors(queryCount, dimension, seed: 5678);
        var index = new HnswIndex(dimension, metric, m: 24, efConstruction: 300, seed: 42)
        {
            Ef = 220,
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i, vectors[i]);
        }

        double found = 0;
        foreach (float[] query in queries)
        {
            HashSet<long> exact = BruteForce(vectors, query, metric, k).Select(r => r.Id).ToHashSet();
            IReadOnlyList<(long Id, float Distance)> actual = index.Search(query, k);
            found += actual.Count(r => exact.Contains(r.Id));
        }

        double recall = found / (queryCount * k);
        Assert.True(recall >= 0.95, $"{metric} recall@{k}: {recall:P2}");
    }

    [Theory]
    [InlineData(DistanceMetric.Cosine)]
    [InlineData(DistanceMetric.EuclideanL2)]
    [InlineData(DistanceMetric.DotProduct)]
    public void TinyDataFindsExactNearestNeighbor(DistanceMetric metric)
    {
        float[][] vectors =
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [-1, 0, 0],
        ];
        float[] query = [0.9f, 0.2f, 0];
        var index = new HnswIndex(3, metric, m: 4, efConstruction: 20, seed: 7)
        {
            Ef = 20,
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i + 10, vectors[i]);
        }

        long expected = BruteForce(vectors, query, metric, k: 1, idOffset: 10)[0].Id;
        Assert.Equal(expected, index.Search(query, 1)[0].Id);
    }

    [Fact]
    public void SaveLoadRoundTripProducesIdenticalResults()
    {
        const int count = 300;
        const int dimension = 32;
        float[][] vectors = RandomVectors(count, dimension, seed: 222);
        float[][] queries = RandomVectors(8, dimension, seed: 333);
        var index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 16, efConstruction: 120, seed: 44)
        {
            Ef = 80,
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(10_000 + i, vectors[i]);
        }

        using var stream = new MemoryStream();
        index.Save(stream);
        stream.Position = 0;
        HnswIndex loaded = HnswIndex.Load(stream);

        Assert.Equal(index.Count, loaded.Count);
        foreach (float[] query in queries)
        {
            Assert.Equal(index.Search(query, 12), loaded.Search(query, 12));
        }
    }

    [Fact]
    public void ExportItemsRebuildsPortableIndexWithStoredVectors()
    {
        const int count = 400;
        const int dimension = 32;
        float[][] vectors = RandomVectors(count, dimension, seed: 444);
        float[][] queries = RandomVectors(12, dimension, seed: 555);
        var index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 16, efConstruction: 120, seed: 66)
        {
            Ef = 90,
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(50_000 + i, vectors[i]);
        }

        // Cosine stores unit-normalized vectors, so that is what is exported.
        float[] expected0 = Normalize(vectors[0]);
        (long Id, float[] Vector)[] exported = index.ExportItems().ToArray();
        Assert.Equal(count, exported.Length);
        Assert.Equal(expected0, exported[0].Vector);
        exported[0].Vector[0] = 12345;
        Assert.Equal(expected0[0], index.ExportItems().First().Vector[0]);

        using var stream = new MemoryStream();
        index.Save(stream);
        stream.Position = 0;
        HnswIndex loaded = HnswIndex.Load(stream);
        Assert.Equal(expected0, loaded.ExportItems().First().Vector);

        HnswIndex rebuilt = HnswIndex.Build(
            dimension,
            DistanceMetric.Cosine,
            index.ExportItems(),
            m: 16,
            efConstruction: 120,
            ef: 90,
            seed: 66);

        Assert.Equal(index.Count, rebuilt.Count);
        foreach (float[] query in queries)
        {
            Assert.Equal(index.Search(query, 10).Select(r => r.Id), rebuilt.Search(query, 10).Select(r => r.Id));
        }
    }

    [Fact]
    public void LoadsLegacyV3FormatAndDiscardsDuplicateVector()
    {
        // Hand-craft a version-3 stream: a single DotProduct node (stored == original) so the loader
        // must read and discard the second vector copy that older formats persisted.
        const uint magic = 0x31575348;
        float[] vector = [1f, 2f, 3f];
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(magic);
            writer.Write(3);                                 // version
            writer.Write(vector.Length);                     // dimension
            writer.Write((int)DistanceMetric.DotProduct);    // metric
            writer.Write(2);                                 // m
            writer.Write(10);                                // efConstruction
            writer.Write(10);                                // ef
            writer.Write(0);                                 // entryPoint
            writer.Write(0);                                 // maxLevel
            writer.Write(1);                                 // count
            writer.Write(false);                             // allowReplaceDeleted

            writer.Write(7L);                                // id
            writer.Write(0);                                 // level
            writer.Write(false);                             // deleted
            foreach (float f in vector) writer.Write(f);     // stored vector
            foreach (float f in vector) writer.Write(f);     // original vector (discarded on load)
            writer.Write(1);                                 // layer count
            writer.Write(0);                                 // layer 0 link count
        }

        stream.Position = 0;
        HnswIndex loaded = HnswIndex.Load(stream);
        Assert.Equal(1, loaded.Count);
        Assert.True(loaded.TryGetVector(7, out float[] stored));
        Assert.Equal(vector, stored);
        Assert.Equal(7, loaded.Search(vector, 1)[0].Id);
    }

    [Fact]
    public void HandlesEdgeCases()
    {
        var index = new HnswIndex(3, DistanceMetric.EuclideanL2, seed: 1);
        Assert.Empty(index.Search([1, 2, 3], 10));
        Assert.Throws<ArgumentException>(() => index.Search([1, 2], 1));

        index.Add(123, [1, 2, 3]);
        Assert.Throws<ArgumentException>(() => index.Add(123, [1, 2, 3]));
        Assert.Throws<ArgumentException>(() => index.Add(124, [1, 2]));

        IReadOnlyList<(long Id, float Distance)> all = index.Search([1, 2, 3], 10);
        Assert.Single(all);
        Assert.Equal(123, all[0].Id);
    }

    private static float[][] RandomVectors(int count, int dimension, int seed)
    {
        var random = new Random(seed);
        var vectors = new float[count][];
        for (int i = 0; i < vectors.Length; i++)
        {
            vectors[i] = new float[dimension];
            for (int j = 0; j < dimension; j++)
            {
                vectors[i][j] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }

        return vectors;
    }

    private static float[] Normalize(float[] vector)
    {
        var copy = (float[])vector.Clone();
        float norm = MathF.Sqrt(TensorPrimitives.Dot(copy, copy));
        if (norm > 0)
        {
            TensorPrimitives.Divide(copy, norm, copy);
        }

        return copy;
    }

    private static IReadOnlyList<(long Id, float Distance)> BruteForce(
        float[][] vectors,
        float[] query,
        DistanceMetric metric,
        int k,
        long idOffset = 0)
    {
        return vectors
            .Select((vector, index) => (Id: idOffset + index, Distance: Distance(query, vector, metric)))
            .OrderBy(r => r.Distance)
            .ThenBy(r => r.Id)
            .Take(k)
            .ToArray();
    }

    private static float Distance(float[] left, float[] right, DistanceMetric metric) => metric switch
    {
        DistanceMetric.Cosine => 1.0f - Dot(left, right) / (Norm(left) * Norm(right)),
        DistanceMetric.EuclideanL2 => MathF.Sqrt(left.Zip(right).Sum(p => (p.First - p.Second) * (p.First - p.Second))),
        DistanceMetric.DotProduct => -Dot(left, right),
        _ => throw new InvalidOperationException(),
    };

    private static float Dot(float[] left, float[] right) => left.Zip(right).Sum(p => p.First * p.Second);

    private static float Norm(float[] vector) => MathF.Sqrt(Dot(vector, vector));
}
