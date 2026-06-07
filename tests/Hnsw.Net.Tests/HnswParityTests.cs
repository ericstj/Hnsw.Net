using System.Text.Json;
using HnswNet;
using Xunit;
using Xunit.Abstractions;

namespace Hnsw.Net.Tests;

public sealed class HnswParityTests
{
    private readonly ITestOutputHelper _output;

    public HnswParityTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(DistanceMetric.Cosine)]
    [InlineData(DistanceMetric.EuclideanL2)]
    [InlineData(DistanceMetric.DotProduct)]
    public void MatchesHnswlibOracleQuality(DistanceMetric metric)
    {
        ParityOracle oracle = LoadOracle();
        (float[][] vectors, float[][] queries) = LoadVectors(oracle);
        MetricOracle metricOracle = oracle.Metrics[metric.ToString()];

        var index = new HnswIndex(
            oracle.Dimension,
            metric,
            oracle.Parameters.M,
            oracle.Parameters.EfConstruction,
            oracle.Parameters.HnswlibSeed)
        {
            Ef = oracle.Parameters.Ef,
        };

        for (int i = 0; i < vectors.Length; i++)
        {
            index.Add(i, vectors[i]);
        }

        double oursFound = 0;
        double hnswlibFound = 0;
        double overlapFound = 0;
        for (int i = 0; i < queries.Length; i++)
        {
            HashSet<long> exact = metricOracle.ExactIds[i].Select(id => (long)id).ToHashSet();
            HashSet<long> hnswlib = metricOracle.HnswlibIds[i].Select(id => (long)id).ToHashSet();
            HashSet<long> ours = index.Search(queries[i], oracle.K).Select(r => r.Id).ToHashSet();

            oursFound += ours.Count(exact.Contains);
            hnswlibFound += hnswlib.Count(exact.Contains);
            overlapFound += ours.Count(hnswlib.Contains);
        }

        double denominator = oracle.QueryCount * oracle.K;
        double ourRecall = oursFound / denominator;
        double hnswlibRecall = hnswlibFound / denominator;
        double overlap = overlapFound / denominator;
        _output.WriteLine(
            $"{metric}: our recall@{oracle.K}={ourRecall:P2}, hnswlib recall@{oracle.K}={hnswlibRecall:P2}, overlap={overlap:P2}");

        Assert.True(ourRecall >= 0.95, $"{metric} recall@{oracle.K}: {ourRecall:P2}");
        Assert.True(
            ourRecall + 0.03 >= hnswlibRecall,
            $"{metric} recall@{oracle.K}: ours {ourRecall:P2}, hnswlib {hnswlibRecall:P2}");
        Assert.True(overlap >= 0.90, $"{metric} overlap@{oracle.K}: {overlap:P2}");
    }

    private static ParityOracle LoadOracle()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "oracle_parity.json");
        using FileStream stream = File.OpenRead(path);
        return JsonSerializer.Deserialize<ParityOracle>(stream, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidDataException("Could not read parity oracle.");
    }

    private static (float[][] Vectors, float[][] Queries) LoadVectors(ParityOracle oracle)
    {
        string path = Path.Combine(AppContext.BaseDirectory, oracle.DataFile);
        using var reader = new BinaryReader(File.OpenRead(path));
        byte[] magic = reader.ReadBytes(6);
        if (!magic.SequenceEqual("HNPV1\0"u8.ToArray()))
        {
            throw new InvalidDataException("Unexpected parity vector file.");
        }

        int dimension = reader.ReadInt32();
        int vectorCount = reader.ReadInt32();
        int queryCount = reader.ReadInt32();
        Assert.Equal(oracle.Dimension, dimension);
        Assert.Equal(oracle.VectorCount, vectorCount);
        Assert.Equal(oracle.QueryCount, queryCount);

        return (ReadMatrix(reader, vectorCount, dimension), ReadMatrix(reader, queryCount, dimension));
    }

    private static float[][] ReadMatrix(BinaryReader reader, int rowCount, int dimension)
    {
        var matrix = new float[rowCount][];
        for (int i = 0; i < rowCount; i++)
        {
            matrix[i] = new float[dimension];
            for (int j = 0; j < dimension; j++)
            {
                matrix[i][j] = reader.ReadSingle();
            }
        }

        return matrix;
    }

    private sealed class ParityOracle
    {
        public string DataFile { get; set; } = "";
        public int Dimension { get; set; }
        public int VectorCount { get; set; }
        public int QueryCount { get; set; }
        public int K { get; set; }
        public ParityParameters Parameters { get; set; } = new();
        public Dictionary<string, MetricOracle> Metrics { get; set; } = new();
    }

    private sealed class ParityParameters
    {
        public int M { get; set; }
        public int EfConstruction { get; set; }
        public int Ef { get; set; }
        public int HnswlibSeed { get; set; }
    }

    private sealed class MetricOracle
    {
        public int[][] HnswlibIds { get; set; } = [];
        public int[][] ExactIds { get; set; } = [];
    }
}
