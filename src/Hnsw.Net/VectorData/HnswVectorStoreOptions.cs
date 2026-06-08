using Microsoft.Extensions.AI;

namespace HnswNet;

/// <summary>
/// Options for creating an <see cref="HnswVectorStore" />.
/// </summary>
public sealed class HnswVectorStoreOptions
{
    /// <summary>
    /// Gets or sets the default embedding generator used by collections that store a non-vector property
    /// (for example, <see cref="string" />) and rely on automatic embedding generation.
    /// </summary>
    public IEmbeddingGenerator? EmbeddingGenerator { get; set; }

    /// <summary>Gets or sets the default HNSW <c>M</c> parameter for collections created by this store.</summary>
    public int M { get; set; } = 16;

    /// <summary>Gets or sets the default HNSW <c>efConstruction</c> parameter for collections created by this store.</summary>
    public int EfConstruction { get; set; } = 200;

    /// <summary>Gets or sets the default HNSW query-time <c>ef</c> for collections created by this store.</summary>
    public int? Ef { get; set; }

    /// <summary>Gets or sets the random seed used when building HNSW indexes, for reproducible graphs.</summary>
    public int? Seed { get; set; }
}
