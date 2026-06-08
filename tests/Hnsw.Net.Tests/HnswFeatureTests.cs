using HnswNet;
using Xunit;

namespace Hnsw.Net.Tests;

public sealed class HnswFeatureTests
{
    [Fact]
    public void FilterRestrictsResultsToAllowedIds()
    {
        const int count = 1_000;
        const int dimension = 48;
        float[][] vectors = RandomVectors(count, dimension, seed: 11);
        var index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 16, efConstruction: 200, seed: 3)
        {
            Ef = 200,
        };
        for (int i = 0; i < count; i++)
        {
            index.Add(i, vectors[i]);
        }

        bool Even(long id) => id % 2 == 0;
        IReadOnlyList<(long Id, float Distance)> results = index.Search(vectors[7], 10, Even);

        Assert.Equal(10, results.Count);
        Assert.All(results, r => Assert.True(Even(r.Id)));
    }

    [Fact]
    public void FilterAcceptingSingleIdReturnsThatId()
    {
        const int dimension = 16;
        float[][] vectors = RandomVectors(200, dimension, seed: 21);
        var index = new HnswIndex(dimension, DistanceMetric.EuclideanL2, m: 16, efConstruction: 200, seed: 9)
        {
            Ef = 200,
        };
        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i, vectors[i]);
        }

        IReadOnlyList<(long Id, float Distance)> results = index.Search(vectors[0], 5, id => id == 123);
        (long Id, float Distance) only = Assert.Single(results);
        Assert.Equal(123, only.Id);
    }

    [Fact]
    public void MarkDeletedExcludesFromSearchAndUnmarkRestores()
    {
        const int dimension = 32;
        float[][] vectors = RandomVectors(500, dimension, seed: 31);
        var index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 16, efConstruction: 200, seed: 5)
        {
            Ef = 200,
        };
        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i, vectors[i]);
        }

        long nearest = index.Search(vectors[42], 1)[0].Id;
        index.MarkDeleted(nearest);

        Assert.Equal(1, index.DeletedCount);
        Assert.Equal(vectors.Length - 1, index.ActiveCount);
        Assert.False(index.Contains(nearest));
        Assert.DoesNotContain(index.Search(vectors[42], 10), r => r.Id == nearest);

        index.UnmarkDeleted(nearest);
        Assert.Equal(0, index.DeletedCount);
        Assert.True(index.Contains(nearest));
        Assert.Equal(nearest, index.Search(vectors[42], 1)[0].Id);
    }

    [Fact]
    public void MarkDeletedUnknownIdThrows()
    {
        var index = new HnswIndex(3, DistanceMetric.EuclideanL2, seed: 1);
        index.Add(1, [1, 2, 3]);
        Assert.Throws<KeyNotFoundException>(() => index.MarkDeleted(999));
        Assert.Throws<KeyNotFoundException>(() => index.UnmarkDeleted(999));
    }

    [Fact]
    public void ReplaceDeletedReusesSlotAndIndexesNewVector()
    {
        const int dimension = 32;
        float[][] vectors = RandomVectors(400, dimension, seed: 41);
        var index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 16, efConstruction: 200, seed: 8, allowReplaceDeleted: true)
        {
            Ef = 200,
        };
        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i, vectors[i]);
        }

        int slotsBefore = index.Count;

        // Delete several non-entry ids, then re-add new ones. Slots should be reused (Count unchanged).
        for (long id = 100; id < 110; id++)
        {
            index.MarkDeleted(id);
        }

        float[][] replacements = RandomVectors(10, dimension, seed: 99);
        for (int i = 0; i < replacements.Length; i++)
        {
            index.Add(1_000 + i, replacements[i]);
        }

        Assert.Equal(slotsBefore, index.Count);
        Assert.Equal(0, index.DeletedCount);

        // Each new vector must be findable; the old deleted ids must not resurface.
        for (int i = 0; i < replacements.Length; i++)
        {
            IReadOnlyList<(long Id, float Distance)> results = index.Search(replacements[i], 1);
            Assert.Equal(1_000 + i, results[0].Id);
        }

        for (long id = 100; id < 110; id++)
        {
            Assert.False(index.Contains(id));
        }
    }

    [Fact]
    public void ConcurrentSearchesMatchSerialResults()
    {
        const int count = 2_000;
        const int dimension = 48;
        float[][] vectors = RandomVectors(count, dimension, seed: 51);
        float[][] queries = RandomVectors(200, dimension, seed: 52);
        var index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 16, efConstruction: 200, seed: 6)
        {
            Ef = 120,
        };
        for (int i = 0; i < count; i++)
        {
            index.Add(i, vectors[i]);
        }

        long[][] serial = queries.Select(q => index.Search(q, 10).Select(r => r.Id).ToArray()).ToArray();

        var parallel = new long[queries.Length][];
        Parallel.For(0, queries.Length, i =>
        {
            parallel[i] = index.Search(queries[i], 10).Select(r => r.Id).ToArray();
        });

        for (int i = 0; i < queries.Length; i++)
        {
            Assert.Equal(serial[i], parallel[i]);
        }
    }

    [Fact]
    public void SaveLoadRoundTripsDeletedState()
    {
        const int dimension = 32;
        float[][] vectors = RandomVectors(300, dimension, seed: 61);
        var index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 16, efConstruction: 120, seed: 7, allowReplaceDeleted: true)
        {
            Ef = 80,
        };
        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i, vectors[i]);
        }

        for (long id = 5; id < 15; id++)
        {
            index.MarkDeleted(id);
        }

        using var stream = new MemoryStream();
        index.Save(stream);
        stream.Position = 0;
        HnswIndex loaded = HnswIndex.Load(stream);

        Assert.Equal(index.Count, loaded.Count);
        Assert.Equal(index.DeletedCount, loaded.DeletedCount);
        Assert.True(loaded.AllowReplaceDeleted);
        for (long id = 5; id < 15; id++)
        {
            Assert.False(loaded.Contains(id));
        }

        float[][] queries = RandomVectors(8, dimension, seed: 62);
        foreach (float[] query in queries)
        {
            Assert.Equal(index.Search(query, 10), loaded.Search(query, 10));
        }
    }

    [Theory]
    [InlineData(DistanceMetric.Cosine)]
    [InlineData(DistanceMetric.EuclideanL2)]
    [InlineData(DistanceMetric.DotProduct)]
    public void BruteForceReturnsExactNearestNeighbors(DistanceMetric metric)
    {
        const int count = 500;
        const int dimension = 24;
        float[][] vectors = RandomVectors(count, dimension, seed: 71);
        float[][] queries = RandomVectors(20, dimension, seed: 72);
        var index = new BruteForceIndex(dimension, metric);
        for (int i = 0; i < count; i++)
        {
            index.Add(i, vectors[i]);
        }

        Assert.Equal(count, index.Count);
        foreach (float[] query in queries)
        {
            long[] expected = Reference(vectors, query, metric, 5).Select(r => r.Id).ToArray();
            long[] actual = index.Search(query, 5).Select(r => r.Id).ToArray();
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void BruteForceSupportsRemovalAndFiltering()
    {
        const int dimension = 16;
        float[][] vectors = RandomVectors(100, dimension, seed: 81);
        var index = new BruteForceIndex(dimension, DistanceMetric.EuclideanL2);
        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i, vectors[i]);
        }

        long nearest = index.Search(vectors[10], 1)[0].Id;
        Assert.True(index.Remove(nearest));
        Assert.False(index.Contains(nearest));
        Assert.Equal(vectors.Length - 1, index.Count);
        Assert.DoesNotContain(index.Search(vectors[10], 5), r => r.Id == nearest);

        IReadOnlyList<(long Id, float Distance)> filtered = index.Search(vectors[0], 5, id => id == 50);
        Assert.Equal(50, Assert.Single(filtered).Id);
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

    private static IReadOnlyList<(long Id, float Distance)> Reference(
        float[][] vectors,
        float[] query,
        DistanceMetric metric,
        int k)
    {
        return vectors
            .Select((vector, index) => (Id: (long)index, Distance: Distance(query, vector, metric)))
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
