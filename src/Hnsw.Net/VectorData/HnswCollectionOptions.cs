using Microsoft.Extensions.VectorData;

namespace HnswNet;

/// <summary>
/// Options for creating an <see cref="HnswCollection{TKey, TRecord}" />.
/// </summary>
public sealed class HnswCollectionOptions : VectorStoreCollectionOptions
{
    /// <summary>Gets or sets the HNSW <c>M</c> parameter (number of bi-directional links per node).</summary>
    public int M { get; set; } = 16;

    /// <summary>Gets or sets the HNSW <c>efConstruction</c> parameter (build-time candidate list size).</summary>
    public int EfConstruction { get; set; } = 200;

    /// <summary>Gets or sets the HNSW query-time <c>ef</c> (search candidate list size). Falls back to the index default when unset.</summary>
    public int? Ef { get; set; }

    /// <summary>Gets or sets the random seed used when building the HNSW index, for reproducible graphs.</summary>
    public int? Seed { get; set; }
}
