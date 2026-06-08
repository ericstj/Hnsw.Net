namespace HnswNet;

/// <summary>
/// Mutable storage shared by every <see cref="HnswCollection{TKey, TRecord}" /> instance that targets the same
/// collection name within a store. Holds the HNSW index, the record payloads, and the mapping between the
/// caller's keys and the <see cref="long" /> ids used by <see cref="HnswIndex" />.
/// </summary>
internal sealed class HnswCollectionData
{
    public readonly object Lock = new();

    public HnswIndex? Index;

    public int Dimension;

    /// <summary>Record payloads keyed by the (boxed) caller key.</summary>
    public readonly Dictionary<object, Entry> Records = new();

    /// <summary>Maps an HNSW id back to the caller key.</summary>
    public readonly Dictionary<long, object> IdToKey = new();

    public long NextId;

    public sealed class Entry
    {
        public required object Record { get; init; }

        public required long Id { get; init; }

        public required ReadOnlyMemory<float> Vector { get; init; }
    }
}
