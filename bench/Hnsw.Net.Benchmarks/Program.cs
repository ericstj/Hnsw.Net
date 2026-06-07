using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using HnswNet;

namespace HnswNet.Benchmarks;

[MemoryDiagnoser]
public class BuildBenchmarks
{
    private float[][] _vectors = null!;

    [Params(2_000)]
    public int Count { get; set; }

    [Params(64)]
    public int Dimension { get; set; }

    [GlobalSetup]
    public void Setup() => _vectors = RandomVectors(Count, Dimension, seed: 1234);

    [Benchmark]
    public int AddAll()
    {
        var index = new HnswIndex(Dimension, DistanceMetric.Cosine, m: 24, efConstruction: 300, seed: 42);
        for (int i = 0; i < _vectors.Length; i++)
        {
            index.Add(i, _vectors[i]);
        }

        return index.Count;
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
}

[MemoryDiagnoser]
public class SearchBenchmarks
{
    private HnswIndex _index = null!;
    private float[] _query = null!;

    [Params(10)]
    public int K { get; set; }

    [Params(220)]
    public int Ef { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        const int count = 2_000;
        const int dimension = 64;
        float[][] vectors = RandomVectors(count, dimension, seed: 1234);
        _query = RandomVectors(1, dimension, seed: 5678)[0];
        _index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 24, efConstruction: 300, seed: 42)
        {
            Ef = Ef,
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            _index.Add(i, vectors[i]);
        }
    }

    [Benchmark]
    public IReadOnlyList<(long Id, float Distance)> SearchSingle() => _index.Search(_query, K);

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
}

[MemoryDiagnoser]
public class PersistenceBenchmarks
{
    private HnswIndex _index = null!;
    private byte[] _saved = null!;

    [GlobalSetup]
    public void Setup()
    {
        const int count = 2_000;
        const int dimension = 64;
        float[][] vectors = RandomVectors(count, dimension, seed: 1234);
        _index = new HnswIndex(dimension, DistanceMetric.Cosine, m: 24, efConstruction: 300, seed: 42)
        {
            Ef = 220,
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            _index.Add(i, vectors[i]);
        }

        using var stream = new MemoryStream();
        _index.Save(stream);
        _saved = stream.ToArray();
    }

    [Benchmark]
    public long Save()
    {
        using var stream = new MemoryStream();
        _index.Save(stream);
        return stream.Length;
    }

    [Benchmark]
    public HnswIndex Load()
    {
        using var stream = new MemoryStream(_saved, writable: false);
        return HnswIndex.Load(stream);
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
}

public static class Program
{
    public static void Main(string[] args) =>
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
}
