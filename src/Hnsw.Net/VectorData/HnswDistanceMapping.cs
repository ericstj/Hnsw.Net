using Microsoft.Extensions.VectorData;

namespace HnswNet;

/// <summary>
/// Maps Microsoft.Extensions.VectorData distance functions onto the HNSW <see cref="DistanceMetric" />
/// and converts raw HNSW distances into the score convention each distance function implies.
/// </summary>
internal static class HnswDistanceMapping
{
    public static DistanceMetric ToMetric(string? distanceFunction) => distanceFunction switch
    {
        null
            or DistanceFunction.CosineSimilarity
            or DistanceFunction.CosineDistance => DistanceMetric.Cosine,
        DistanceFunction.DotProductSimilarity
            or DistanceFunction.NegativeDotProductSimilarity => DistanceMetric.DotProduct,
        DistanceFunction.EuclideanDistance => DistanceMetric.EuclideanL2,
        _ => throw new NotSupportedException($"The distance function '{distanceFunction}' is not supported by the HNSW connector."),
    };

    /// <summary>
    /// Converts a raw HNSW distance (lower is always nearer) into the score expected for the distance function.
    /// </summary>
    public static float ToScore(float distance, string? distanceFunction) => distanceFunction switch
    {
        // HNSW cosine distance is 1 - cosine similarity.
        null or DistanceFunction.CosineSimilarity => 1f - distance,
        DistanceFunction.CosineDistance => distance,

        // HNSW dot-product distance is the negative inner product.
        DistanceFunction.DotProductSimilarity => -distance,
        DistanceFunction.NegativeDotProductSimilarity => distance,

        DistanceFunction.EuclideanDistance => distance,
        _ => throw new NotSupportedException($"The distance function '{distanceFunction}' is not supported by the HNSW connector."),
    };

    /// <summary>
    /// Indicates whether a higher score means a closer match for the distance function, which determines how
    /// <see cref="VectorSearchOptions{TRecord}.ScoreThreshold" /> is applied.
    /// </summary>
    public static bool HigherScoreIsCloser(string? distanceFunction) => distanceFunction switch
    {
        null
            or DistanceFunction.CosineSimilarity
            or DistanceFunction.DotProductSimilarity => true,
        DistanceFunction.CosineDistance
            or DistanceFunction.NegativeDotProductSimilarity
            or DistanceFunction.EuclideanDistance => false,
        _ => throw new NotSupportedException($"The distance function '{distanceFunction}' is not supported by the HNSW connector."),
    };
}
