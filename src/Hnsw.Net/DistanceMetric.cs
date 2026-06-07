namespace HnswNet;

/// <summary>Distance metric used by <see cref="HnswIndex" />.</summary>
public enum DistanceMetric
{
    /// <summary>Cosine distance, computed as <c>1 - cosine similarity</c>.</summary>
    Cosine,

    /// <summary>Euclidean L2 distance.</summary>
    EuclideanL2,

    /// <summary>Negative inner product; lower values are more similar.</summary>
    DotProduct,
}
