using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace HnswNet;

/// <summary>
/// An HNSW approximate-nearest-neighbor index over <see cref="float" /> vectors.
/// Builds and modifications are serialized; searches are thread-safe and may run concurrently.
/// </summary>
public sealed class HnswIndex : IDisposable
{
    private const int FormatVersion = 4;
    private const uint Magic = 0x31575348; // HSW1, little-endian.

    private readonly List<Node> _nodes = new();
    private readonly Dictionary<long, int> _ids = new();
    private readonly Random _random;
    private readonly double _levelMultiplier;
    private readonly bool _allowReplaceDeleted;
    private readonly Stack<int> _deletedSlots = new();
    private int _deletedCount;
    private int _entryPoint = -1;
    private int _maxLevel = -1;

    private readonly ReaderWriterLockSlim _lock = new();
    private readonly ConcurrentBag<Scratch> _scratchPool = new();
    private readonly Comparison<Candidate> _candidateComparison;
    private VectorBlock _vectors;
    private bool _disposed;

    /// <summary>Initializes a new HNSW index.</summary>
    /// <param name="dimension">Vector dimension. All indexed and query vectors must have this length.</param>
    /// <param name="metric">Distance metric. Dot product uses negative inner product so lower is better.</param>
    /// <param name="m">Maximum number of connections per non-zero layer. Layer 0 uses <c>2 * m</c>.</param>
    /// <param name="efConstruction">Beam width used while adding vectors.</param>
    /// <param name="seed">Optional random seed for reproducible layer assignment.</param>
    /// <param name="allowReplaceDeleted">When true, <see cref="Add(long, ReadOnlySpan{float})" /> reuses slots freed by <see cref="MarkDeleted" />.</param>
    public HnswIndex(int dimension, DistanceMetric metric, int m = 16, int efConstruction = 200, int? seed = null, bool allowReplaceDeleted = false)
    {
        if (dimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension));
        }
        if (m <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(m));
        }
        if (efConstruction < m)
        {
            throw new ArgumentOutOfRangeException(nameof(efConstruction), "efConstruction must be at least m.");
        }

        Dimension = dimension;
        Metric = metric;
        M = m;
        EfConstruction = efConstruction;
        Ef = 50;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _levelMultiplier = 1.0 / Math.Log(m);
        _allowReplaceDeleted = allowReplaceDeleted;
        _candidateComparison = CompareCandidates;
        _vectors = new HeapVectorBlock(dimension);
    }

    private HnswIndex(int dimension, DistanceMetric metric, int m, int efConstruction, int ef, int entryPoint, int maxLevel, bool allowReplaceDeleted)
    {
        Dimension = dimension;
        Metric = metric;
        M = m;
        EfConstruction = efConstruction;
        Ef = ef;
        _entryPoint = entryPoint;
        _maxLevel = maxLevel;
        _random = new Random(0);
        _levelMultiplier = 1.0 / Math.Log(m);
        _allowReplaceDeleted = allowReplaceDeleted;
        _candidateComparison = CompareCandidates;
        _vectors = new HeapVectorBlock(dimension);
    }

    /// <summary>Gets the vector dimension.</summary>
    public int Dimension { get; }

    /// <summary>Gets the distance metric.</summary>
    public DistanceMetric Metric { get; }

    /// <summary>Gets the maximum number of connections per non-zero layer.</summary>
    public int M { get; }

    /// <summary>Gets the construction beam width.</summary>
    public int EfConstruction { get; }

    /// <summary>Gets or sets the search beam width.</summary>
    public int Ef { get; set; }

    /// <summary>Gets the number of allocated slots, including those freed by <see cref="MarkDeleted" />.</summary>
    public int Count
    {
        get { _lock.EnterReadLock(); try { return _nodes.Count; } finally { _lock.ExitReadLock(); } }
    }

    /// <summary>Gets the number of slots currently marked deleted.</summary>
    public int DeletedCount
    {
        get { _lock.EnterReadLock(); try { return _deletedCount; } finally { _lock.ExitReadLock(); } }
    }

    /// <summary>Gets the number of live (non-deleted) vectors returnable from a search.</summary>
    public int ActiveCount
    {
        get { _lock.EnterReadLock(); try { return _nodes.Count - _deletedCount; } finally { _lock.ExitReadLock(); } }
    }

    /// <summary>Gets whether <see cref="Add(long, ReadOnlySpan{float})" /> reuses slots freed by <see cref="MarkDeleted" />.</summary>
    public bool AllowReplaceDeleted => _allowReplaceDeleted;

    /// <summary>
    /// Returns copies of the live ids and their stored vectors, suitable for rebuilding a portable
    /// index. For <see cref="DistanceMetric.Cosine" /> the stored vector is unit-normalized, matching
    /// what the index searches against; re-adding it reproduces the same index.
    /// </summary>
    public IEnumerable<(long Id, float[] Vector)> ExportItems()
    {
        _lock.EnterReadLock();
        try
        {
            var items = new List<(long, float[])>(_nodes.Count - _deletedCount);
            for (int slot = 0; slot < _nodes.Count; slot++)
            {
                Node node = _nodes[slot];
                if (!node.Deleted)
                {
                    items.Add((node.Id, _vectors.Vector(slot).ToArray()));
                }
            }

            return items;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>Builds an index by adding all exported ids and vectors with the supplied parameters.</summary>
    public static HnswIndex Build(
        int dimension,
        DistanceMetric metric,
        IEnumerable<(long Id, float[] Vector)> items,
        int m = 16,
        int efConstruction = 200,
        int ef = 50,
        int? seed = null,
        bool allowReplaceDeleted = false)
    {
        ArgumentNullException.ThrowIfNull(items);
        var index = new HnswIndex(dimension, metric, m, efConstruction, seed, allowReplaceDeleted)
        {
            Ef = ef,
        };

        foreach ((long id, float[] vector) in items)
        {
            ArgumentNullException.ThrowIfNull(vector);
            index.Add(id, vector);
        }

        return index;
    }

    /// <summary>
    /// Adds a vector with the specified id. Duplicate ids throw <see cref="ArgumentException" />. When
    /// <see cref="AllowReplaceDeleted" /> is set, a slot previously freed by <see cref="MarkDeleted" /> is reused.
    /// </summary>
    public void Add(long id, ReadOnlySpan<float> vector)
    {
        ValidateVector(vector);
        ThrowIfReadOnly();

        _lock.EnterWriteLock();
        try
        {
            if (_ids.ContainsKey(id))
            {
                throw new ArgumentException("An item with the same id already exists.", nameof(id));
            }

            while (_allowReplaceDeleted && _deletedSlots.Count > 0)
            {
                int slot = _deletedSlots.Pop();
                if (_nodes[slot].Deleted && slot != _entryPoint)
                {
                    ReplaceSlot(slot, id, vector);
                    return;
                }
            }

            int level = RandomLevel();
            int newIndex = _nodes.Count;
            EnsureStoredCapacity(newIndex + 1);
            PrepareVectorInto(vector, StoredSpanMutable(newIndex));
            var node = new Node(id, level);
            _nodes.Add(node);
            _ids.Add(id, newIndex);

            if (_entryPoint < 0)
            {
                _entryPoint = newIndex;
                _maxLevel = level;
                return;
            }

            Scratch s = RentScratch();
            try
            {
                LinkNode(newIndex, level, s);
            }
            finally
            {
                ReturnScratch(s);
            }

            if (level > _maxLevel)
            {
                _entryPoint = newIndex;
                _maxLevel = level;
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>Adds a vector with the specified id.</summary>
    public void Add(long id, ReadOnlyMemory<float> vector) => Add(id, vector.Span);

    /// <summary>
    /// Marks the vector with the specified id deleted. Deleted vectors stay in the graph for connectivity but are
    /// never returned from a search. Throws <see cref="KeyNotFoundException" /> if the id is unknown.
    /// </summary>
    public void MarkDeleted(long id)
    {
        ThrowIfReadOnly();
        _lock.EnterWriteLock();
        try
        {
            if (!_ids.TryGetValue(id, out int slot))
            {
                throw new KeyNotFoundException($"No item with id {id} exists.");
            }

            Node node = _nodes[slot];
            if (node.Deleted)
            {
                return;
            }

            node.Deleted = true;
            _deletedCount++;
            if (_allowReplaceDeleted && slot != _entryPoint)
            {
                _deletedSlots.Push(slot);
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>Restores a vector previously marked by <see cref="MarkDeleted" />, making it searchable again.</summary>
    public void UnmarkDeleted(long id)
    {
        ThrowIfReadOnly();
        _lock.EnterWriteLock();
        try
        {
            if (!_ids.TryGetValue(id, out int slot))
            {
                throw new KeyNotFoundException($"No item with id {id} exists.");
            }

            Node node = _nodes[slot];
            if (!node.Deleted)
            {
                return;
            }

            node.Deleted = false;
            _deletedCount--;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>Returns whether a live (non-deleted) vector with the specified id exists.</summary>
    public bool Contains(long id)
    {
        _lock.EnterReadLock();
        try
        {
            return _ids.TryGetValue(id, out int slot) && !_nodes[slot].Deleted;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets a copy of the stored vector for a live id. For <see cref="DistanceMetric.Cosine" /> this
    /// is the unit-normalized vector. Returns false for unknown or deleted ids.
    /// </summary>
    public bool TryGetVector(long id, out float[] vector)
    {
        _lock.EnterReadLock();
        try
        {
            if (_ids.TryGetValue(id, out int slot) && !_nodes[slot].Deleted)
            {
                vector = _vectors.Vector(slot).ToArray();
                return true;
            }

            vector = Array.Empty<float>();
            return false;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    private void LinkNode(int newIndex, int level, Scratch s)
    {
        int entryPoint = _entryPoint;
        float entryDistance = Distance(StoredSpan(newIndex), StoredSpan(entryPoint));

        for (int layer = _maxLevel; layer > level; layer--)
        {
            (entryPoint, entryDistance) = SearchGreedy(StoredSpan(newIndex), entryPoint, entryDistance, layer);
        }

        Node node = _nodes[newIndex];
        int topLayer = Math.Min(level, _maxLevel);
        for (int layer = topLayer; layer >= 0; layer--)
        {
            SearchLayer(StoredSpan(newIndex), entryPoint, EfConstruction, layer, s, null);
            SelectNeighbors(s.LayerResult, MaxConnections(layer), s.AddSelected, s);
            foreach (int neighbor in s.AddSelected)
            {
                if (neighbor == newIndex)
                {
                    continue;
                }

                node.Links[layer].Add(neighbor);
                List<int> links = _nodes[neighbor].Links[layer];
                if (!links.Contains(newIndex))
                {
                    links.Add(newIndex);
                }
                PruneConnections(neighbor, layer, s);
            }

            if (s.LayerResult.Count > 0)
            {
                Candidate nearest = s.LayerResult[0];
                entryPoint = nearest.Index;
                entryDistance = nearest.Distance;
            }
        }
    }

    private void ReplaceSlot(int slot, long id, ReadOnlySpan<float> vector)
    {
        Node node = _nodes[slot];
        for (int layer = 0; layer < node.Links.Length; layer++)
        {
            foreach (int neighbor in node.Links[layer])
            {
                _nodes[neighbor].Links[layer].Remove(slot);
            }

            node.Links[layer].Clear();
        }

        _ids.Remove(node.Id);
        node.Id = id;
        node.Deleted = false;
        _deletedCount--;
        _ids.Add(id, slot);
        PrepareVectorInto(vector, StoredSpanMutable(slot));

        Scratch s = RentScratch();
        try
        {
            LinkNode(slot, node.Level, s);
        }
        finally
        {
            ReturnScratch(s);
        }
    }

    /// <summary>
    /// Searches for the nearest vectors. Results are sorted by ascending distance; for dot product this means most similar first.
    /// </summary>
    public IReadOnlyList<(long Id, float Distance)> Search(ReadOnlySpan<float> query, int k) => Search(query, k, null);

    /// <summary>
    /// Searches for the nearest vectors, optionally restricted to ids accepted by <paramref name="filter" />.
    /// Deleted vectors are always excluded. The call is thread-safe with respect to concurrent searches.
    /// </summary>
    public IReadOnlyList<(long Id, float Distance)> Search(ReadOnlySpan<float> query, int k, Func<long, bool>? filter)
    {
        ValidateVector(query);
        if (k <= 0)
        {
            return Array.Empty<(long Id, float Distance)>();
        }

        float[] preparedQuery = PrepareVector(query);
        _lock.EnterReadLock();
        try
        {
            if (_entryPoint < 0)
            {
                return Array.Empty<(long Id, float Distance)>();
            }

            Func<int, bool>? allowed = null;
            if (filter is not null)
            {
                allowed = i => !_nodes[i].Deleted && filter(_nodes[i].Id);
            }
            else if (_deletedCount > 0)
            {
                allowed = i => !_nodes[i].Deleted;
            }

            Scratch s = RentScratch();
            try
            {
                int entryPoint = _entryPoint;
                float entryDistance = Distance(preparedQuery, SlotVector(entryPoint));
                for (int layer = _maxLevel; layer > 0; layer--)
                {
                    (entryPoint, entryDistance) = SearchGreedy(preparedQuery, entryPoint, entryDistance, layer);
                }

                int ef = Math.Max(Ef, k);
                SearchLayer(preparedQuery, entryPoint, ef, 0, s, allowed);
                int resultCount = Math.Min(k, s.LayerResult.Count);
                var results = new (long Id, float Distance)[resultCount];
                for (int i = 0; i < resultCount; i++)
                {
                    Candidate candidate = s.LayerResult[i];
                    results[i] = (_nodes[candidate.Index].Id, candidate.Distance);
                }

                return results;
            }
            finally
            {
                ReturnScratch(s);
            }
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>Saves the index to a stream using a versioned binary format.</summary>
    public void Save(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        _lock.EnterReadLock();
        try
        {
            using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            writer.Write(Magic);
            writer.Write(FormatVersion);
            writer.Write(Dimension);
            writer.Write((int)Metric);
            writer.Write(M);
            writer.Write(EfConstruction);
            writer.Write(Ef);
            writer.Write(_entryPoint);
            writer.Write(_maxLevel);
            writer.Write(_nodes.Count);
            writer.Write(_allowReplaceDeleted);

            // Vector section: one normalized vector per slot, contiguous and fixed-stride, so the
            // read path can memory-map it and address slot s at base + s * Dimension * sizeof(float).
            for (int n = 0; n < _nodes.Count; n++)
            {
                WriteFloatsLittleEndian(writer, StoredSpan(n));
            }

            // Graph section: per-node metadata and link lists, kept separate from the vectors.
            for (int n = 0; n < _nodes.Count; n++)
            {
                Node node = _nodes[n];
                writer.Write(node.Id);
                writer.Write(node.Level);
                writer.Write(node.Deleted);
                writer.Write(node.Links.Length);
                for (int layer = 0; layer < node.Links.Length; layer++)
                {
                    writer.Write(node.Links[layer].Count);
                    foreach (int neighbor in node.Links[layer])
                    {
                        writer.Write(neighbor);
                    }
                }
            }
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>Loads an index saved by <see cref="Save" /> into managed memory.</summary>
    public static HnswIndex Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        Header header = ReadHeader(reader);
        var index = new HnswIndex(header.Dimension, header.Metric, header.M, header.EfConstruction, header.Ef, header.EntryPoint, header.MaxLevel, header.AllowReplaceDeleted);
        index._vectors.EnsureCapacity(header.Count);

        if (header.Version <= 3)
        {
            ReadInterleavedBody(reader, index, header);
        }
        else
        {
            // Vector section first (one vector per slot), then the graph section.
            for (int i = 0; i < header.Count; i++)
            {
                ReadFloatsLittleEndian(reader, index._vectors.VectorMutable(i));
            }

            for (int i = 0; i < header.Count; i++)
            {
                long id = reader.ReadInt64();
                int level = reader.ReadInt32();
                bool deleted = reader.ReadBoolean();
                ReadNodeLinks(reader, index, i, id, level, deleted, header);
            }
        }

        return index;
    }

    /// <summary>
    /// Loads an index from a file, memory-mapping the vector section instead of reading it into the
    /// managed heap. The graph is still loaded into memory. The returned index is read-only (calls to
    /// <see cref="Add(long, ReadOnlySpan{float})" /> throw) and owns the mapping, so it must be disposed.
    /// Only the current on-disk format is supported; re-save older indexes first.
    /// </summary>
    public static HnswIndex LoadMapped(string path) => LoadMapped(path, 0);

    /// <summary>
    /// Loads an index that begins at <paramref name="baseOffset" /> within a larger file, memory-mapping
    /// its vector section. Use this when an index is embedded in a container format (for example a
    /// collection snapshot). Behaves like <see cref="LoadMapped(string)" /> otherwise: the result is
    /// read-only, owns the mapping, and must be disposed.
    /// </summary>
    public static HnswIndex LoadMapped(string path, long baseOffset)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentOutOfRangeException.ThrowIfNegative(baseOffset);

        // The mapped vector section is read as host-endian floats straight off the pages, so it is
        // only correct on little-endian platforms (the only ones .NET supports). Reject otherwise
        // rather than return silently wrong results; the stream loader handles big-endian hosts.
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("Memory-mapped loading is only supported on little-endian platforms.");
        }

        Header header;
        long vectorOffset;
        HnswIndex index;
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            stream.Seek(baseOffset, SeekOrigin.Begin);
            header = ReadHeader(reader);
            if (header.Version != FormatVersion)
            {
                throw new InvalidDataException("Memory-mapped loading requires the current index format; re-save the index.");
            }

            vectorOffset = stream.Position;
            if (header.Count < 0 || header.Dimension < 0)
            {
                throw new InvalidDataException("The index header specifies a negative count or dimension.");
            }

            long vectorBytes;
            try
            {
                vectorBytes = checked((long)header.Count * header.Dimension * sizeof(float));
            }
            catch (OverflowException)
            {
                throw new InvalidDataException("The index vector section size is invalid.");
            }

            if (vectorOffset + vectorBytes > stream.Length)
            {
                throw new InvalidDataException("The index vector section extends beyond the end of the file.");
            }

            index = new HnswIndex(header.Dimension, header.Metric, header.M, header.EfConstruction, header.Ef, header.EntryPoint, header.MaxLevel, header.AllowReplaceDeleted);

            stream.Seek(vectorOffset + vectorBytes, SeekOrigin.Begin);
            for (int i = 0; i < header.Count; i++)
            {
                long id = reader.ReadInt64();
                int level = reader.ReadInt32();
                bool deleted = reader.ReadBoolean();
                ReadNodeLinks(reader, index, i, id, level, deleted, header);
            }
        }

        MemoryMappedFile? file = null;
        MemoryMappedViewAccessor? view = null;
        try
        {
            file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
            view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            index._vectors = new MappedVectorBlock(file, view, vectorOffset, header.Dimension);
            return index;
        }
        catch
        {
            view?.Dispose();
            file?.Dispose();
            throw;
        }
    }

    private static Header ReadHeader(BinaryReader reader)
    {
        if (reader.ReadUInt32() != Magic)
        {
            throw new InvalidDataException("The stream is not an Hnsw.Net index.");
        }

        int version = reader.ReadInt32();
        if (version is < 1 or > FormatVersion)
        {
            throw new InvalidDataException("Unsupported Hnsw.Net index format version.");
        }

        int dimension = reader.ReadInt32();
        var metric = (DistanceMetric)reader.ReadInt32();
        int m = reader.ReadInt32();
        int efConstruction = reader.ReadInt32();
        int ef = reader.ReadInt32();
        int entryPoint = reader.ReadInt32();
        int maxLevel = reader.ReadInt32();
        int count = reader.ReadInt32();
        bool allowReplaceDeleted = version >= 3 && reader.ReadBoolean();
        return new Header(version, dimension, metric, m, efConstruction, ef, entryPoint, maxLevel, count, allowReplaceDeleted);
    }

    // Reads the pre-v4 layout where each node's vector(s) are interleaved with its graph data.
    private static void ReadInterleavedBody(BinaryReader reader, HnswIndex index, Header header)
    {
        // Formats 2 and 3 stored a second copy of the pre-normalized vector per node; it is read and
        // discarded since only the (normalized) stored vector is retained now.
        float[] discard = header.Version is 2 or 3 ? new float[header.Dimension] : Array.Empty<float>();
        for (int i = 0; i < header.Count; i++)
        {
            long id = reader.ReadInt64();
            int level = reader.ReadInt32();
            bool deleted = header.Version >= 3 && reader.ReadBoolean();
            ReadFloatsLittleEndian(reader, index._vectors.VectorMutable(i));
            if (discard.Length > 0)
            {
                ReadFloatsLittleEndian(reader, discard.AsSpan());
            }

            ReadNodeLinks(reader, index, i, id, level, deleted, header);
        }
    }

    private static void ReadNodeLinks(BinaryReader reader, HnswIndex index, int i, long id, int level, bool deleted, Header header)
    {
        var node = new Node(id, level) { Deleted = deleted };
        int layerCount = reader.ReadInt32();
        if (layerCount != level + 1)
        {
            throw new InvalidDataException("Invalid layer count in Hnsw.Net index.");
        }

        for (int layer = 0; layer < layerCount; layer++)
        {
            int linkCount = reader.ReadInt32();

            // Bound the count by the node total (a slot can have at most Count distinct neighbors)
            // so a corrupt length cannot drive a huge allocation, then validate every neighbor slot.
            // Unchecked indices would later reach MappedVectorBlock, which builds a span directly
            // from a raw pointer with no bounds check, so an out-of-range link could read arbitrary
            // mapped memory or fault the process instead of throwing.
            if ((uint)linkCount > (uint)header.Count)
            {
                throw new InvalidDataException("Invalid link count in Hnsw.Net index.");
            }

            List<int> links = node.Links[layer];
            for (int link = 0; link < linkCount; link++)
            {
                int neighbor = reader.ReadInt32();
                if ((uint)neighbor >= (uint)header.Count)
                {
                    throw new InvalidDataException("Link index out of range in Hnsw.Net index.");
                }

                links.Add(neighbor);
            }
        }

        index._ids.Add(id, i);
        index._nodes.Add(node);
        if (deleted)
        {
            index._deletedCount++;
            if (header.AllowReplaceDeleted && i != header.EntryPoint)
            {
                index._deletedSlots.Push(i);
            }
        }
    }

    private readonly record struct Header(
        int Version,
        int Dimension,
        DistanceMetric Metric,
        int M,
        int EfConstruction,
        int Ef,
        int EntryPoint,
        int MaxLevel,
        int Count,
        bool AllowReplaceDeleted);

    /// <summary>Releases the memory mapping held by an index loaded via <see cref="LoadMapped(string)" />.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        (_vectors as IDisposable)?.Dispose();
        _lock.Dispose();
    }

    private void ThrowIfReadOnly()
    {
        if (_vectors.IsReadOnly)
        {
            throw new InvalidOperationException("This index was loaded with memory-mapped vectors and is read-only. Load it without mapping to modify it.");
        }
    }

    private void ValidateVector(ReadOnlySpan<float> vector)
    {
        if (vector.Length != Dimension)
        {
            throw new ArgumentException($"Expected vector dimension {Dimension}, got {vector.Length}.", nameof(vector));
        }
    }

    private float[] PrepareVector(ReadOnlySpan<float> vector)
    {
        var copy = vector.ToArray();
        if (Metric == DistanceMetric.Cosine)
        {
            float norm = MathF.Sqrt(TensorPrimitives.Dot(copy, copy));
            if (norm > 0)
            {
                TensorPrimitives.Divide(copy, norm, copy);
            }
        }

        return copy;
    }

    private void PrepareVectorInto(ReadOnlySpan<float> source, Span<float> destination)
    {
        source.CopyTo(destination);
        if (Metric == DistanceMetric.Cosine)
        {
            float norm = MathF.Sqrt(TensorPrimitives.Dot(destination, destination));
            if (norm > 0)
            {
                TensorPrimitives.Divide(destination, norm, destination);
            }
        }
    }

    private void EnsureStoredCapacity(int nodeCount) => _vectors.EnsureCapacity(nodeCount);

    private ReadOnlySpan<float> StoredSpan(int index) => _vectors.Vector(index);

    // Per-slot read accessors. The search path goes through these so the backing storage can later
    // be swapped (e.g. a memory-mapped level-0 block) without touching the algorithms.
    private ReadOnlySpan<float> SlotVector(int slot) => StoredSpan(slot);

    private List<int> SlotLinks(int slot, int layer) => _nodes[slot].Links[layer];

    private Span<float> StoredSpanMutable(int index) => _vectors.VectorMutable(index);

    private static void ReadExactInto(BinaryReader reader, Span<byte> buffer)
    {
        int total = 0;
        while (total < buffer.Length)
        {
            int read = reader.Read(buffer.Slice(total));
            if (read == 0)
            {
                throw new EndOfStreamException();
            }

            total += read;
        }
    }

    // The on-disk vector section is little-endian. On little-endian hosts (the only platforms .NET
    // supports) this is a single bulk byte copy; the big-endian fallbacks keep the format portable.
    private static void WriteFloatsLittleEndian(BinaryWriter writer, ReadOnlySpan<float> values)
    {
        if (BitConverter.IsLittleEndian)
        {
            writer.Write(MemoryMarshal.AsBytes(values));
            return;
        }

        Span<byte> scratch = stackalloc byte[sizeof(float)];
        foreach (float value in values)
        {
            BinaryPrimitives.WriteSingleLittleEndian(scratch, value);
            writer.Write(scratch);
        }
    }

    private static void ReadFloatsLittleEndian(BinaryReader reader, Span<float> destination)
    {
        Span<byte> bytes = MemoryMarshal.AsBytes(destination);
        ReadExactInto(reader, bytes);
        if (!BitConverter.IsLittleEndian)
        {
            for (int i = 0; i < bytes.Length; i += sizeof(float))
            {
                bytes.Slice(i, sizeof(float)).Reverse();
            }
        }
    }

    private int RandomLevel()
    {
        double sample = Math.Max(_random.NextDouble(), double.Epsilon);
        return (int)(-Math.Log(sample) * _levelMultiplier);
    }

    private (int Index, float Distance) SearchGreedy(ReadOnlySpan<float> query, int entryPoint, float entryDistance, int layer)
    {
        bool changed;
        do
        {
            changed = false;
            foreach (int neighbor in SlotLinks(entryPoint, layer))
            {
                float distance = Distance(query, SlotVector(neighbor));
                if (distance < entryDistance)
                {
                    entryDistance = distance;
                    entryPoint = neighbor;
                    changed = true;
                }
            }
        }
        while (changed);

        return (entryPoint, entryDistance);
    }

    private void SearchLayer(ReadOnlySpan<float> query, int entryPoint, int ef, int layer, Scratch s, Func<int, bool>? allowed)
    {
        int version = s.NextVersion(_nodes.Count);
        s.Candidates.Clear();
        s.Nearest.Clear();
        s.Visited[entryPoint] = version;

        float entryDistance = Distance(query, SlotVector(entryPoint));
        var entry = new Candidate(entryPoint, entryDistance);
        s.Candidates.Enqueue(entry, entryDistance);
        float lowerBound;
        if (allowed is null || allowed(entryPoint))
        {
            s.Nearest.Enqueue(entry, -entryDistance);
            lowerBound = entryDistance;
        }
        else
        {
            lowerBound = float.MaxValue;
        }

        while (s.Candidates.Count > 0)
        {
            Candidate current = s.Candidates.Dequeue();
            if (current.Distance > lowerBound)
            {
                break;
            }

            foreach (int neighbor in SlotLinks(current.Index, layer))
            {
                if (s.Visited[neighbor] == version)
                {
                    continue;
                }

                s.Visited[neighbor] = version;
                float distance = Distance(query, SlotVector(neighbor));
                if (s.Nearest.Count < ef || distance < lowerBound)
                {
                    var candidate = new Candidate(neighbor, distance);
                    s.Candidates.Enqueue(candidate, distance);
                    if (allowed is null || allowed(neighbor))
                    {
                        s.Nearest.Enqueue(candidate, -distance);
                        if (s.Nearest.Count > ef)
                        {
                            s.Nearest.Dequeue();
                        }
                    }

                    if (s.Nearest.Count > 0)
                    {
                        s.Nearest.TryPeek(out _, out float priority);
                        lowerBound = -priority;
                    }
                }
            }
        }

        s.LayerResult.Clear();
        foreach ((Candidate element, float _) in s.Nearest.UnorderedItems)
        {
            s.LayerResult.Add(element);
        }

        s.LayerResult.Sort(_candidateComparison);
    }

    private void SelectNeighbors(List<Candidate> candidates, int maxConnections, List<int> result, Scratch s)
    {
        result.Clear();
        s.SelectPruned.Clear();
        candidates.Sort(_candidateComparison);
        foreach (Candidate candidate in candidates)
        {
            ReadOnlySpan<float> candidateSpan = SlotVector(candidate.Index);
            bool good = true;
            foreach (int selected in result)
            {
                if (Distance(candidateSpan, SlotVector(selected)) < candidate.Distance)
                {
                    good = false;
                    break;
                }
            }

            if (good)
            {
                result.Add(candidate.Index);
                if (result.Count == maxConnections)
                {
                    return;
                }
            }
            else
            {
                s.SelectPruned.Add(candidate);
            }
        }

        foreach (Candidate candidate in s.SelectPruned)
        {
            if (!result.Contains(candidate.Index))
            {
                result.Add(candidate.Index);
                if (result.Count == maxConnections)
                {
                    break;
                }
            }
        }
    }

    private void PruneConnections(int nodeIndex, int layer, Scratch s)
    {
        List<int> links = _nodes[nodeIndex].Links[layer];
        int maxConnections = MaxConnections(layer);
        if (links.Count <= maxConnections)
        {
            return;
        }

        s.PruneCandidates.Clear();
        foreach (int neighbor in links)
        {
            s.PruneCandidates.Add(new Candidate(neighbor, Distance(StoredSpan(nodeIndex), StoredSpan(neighbor))));
        }

        SelectNeighbors(s.PruneCandidates, maxConnections, s.PruneSelected, s);
        links.Clear();
        links.AddRange(s.PruneSelected);
    }

    private Scratch RentScratch() => _scratchPool.TryTake(out Scratch? s) ? s : new Scratch();

    private void ReturnScratch(Scratch s) => _scratchPool.Add(s);

    private int CompareCandidates(Candidate left, Candidate right)
    {
        int byDistance = left.Distance.CompareTo(right.Distance);
        return byDistance != 0 ? byDistance : _nodes[left.Index].Id.CompareTo(_nodes[right.Index].Id);
    }

    private int MaxConnections(int layer) => layer == 0 ? 2 * M : M;

    private float Distance(ReadOnlySpan<float> left, ReadOnlySpan<float> right) => Metric switch
    {
        DistanceMetric.Cosine => 1.0f - DotProduct(left, right),
        DistanceMetric.EuclideanL2 => TensorPrimitives.Distance(left, right),
        DistanceMetric.DotProduct => -DotProduct(left, right),
        _ => throw new InvalidOperationException("Unknown distance metric."),
    };

    private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int width = Vector<float>.Count;
        int n = a.Length;
        ref float pa = ref MemoryMarshal.GetReference(a);
        ref float pb = ref MemoryMarshal.GetReference(b);
        var acc0 = Vector<float>.Zero;
        var acc1 = Vector<float>.Zero;
        int i = 0;
        for (; i + 2 * width <= n; i += 2 * width)
        {
            acc0 += Vector.LoadUnsafe(ref pa, (nuint)i) * Vector.LoadUnsafe(ref pb, (nuint)i);
            acc1 += Vector.LoadUnsafe(ref pa, (nuint)(i + width)) * Vector.LoadUnsafe(ref pb, (nuint)(i + width));
        }
        for (; i + width <= n; i += width)
        {
            acc0 += Vector.LoadUnsafe(ref pa, (nuint)i) * Vector.LoadUnsafe(ref pb, (nuint)i);
        }

        float sum = Vector.Sum(acc0 + acc1);
        for (; i < n; i++)
        {
            sum += Unsafe.Add(ref pa, i) * Unsafe.Add(ref pb, i);
        }

        return sum;
    }

    // Slot-addressed storage for the normalized search vectors. The read path goes through this
    // abstraction so the backing store can be either the managed heap (build and default load) or a
    // memory-mapped file (LoadMapped).
    private abstract class VectorBlock
    {
        public abstract bool IsReadOnly { get; }

        public abstract void EnsureCapacity(int slotCount);

        public abstract ReadOnlySpan<float> Vector(int slot);

        public abstract Span<float> VectorMutable(int slot);
    }

    // Chunked heap storage. Splitting the data into bounded chunks keeps each backing array well
    // under the .NET array length limit so the index scales to very large repos, and gives every
    // slot a contiguous span.
    private sealed class HeapVectorBlock : VectorBlock
    {
        private readonly int _dimension;
        private readonly int _vectorsPerChunk;
        private readonly List<float[]> _chunks = new();
        private int _capacity;

        public HeapVectorBlock(int dimension)
        {
            _dimension = dimension;

            // Target ~256 MB (64Mi floats) per chunk, at least one vector, and never let a chunk's
            // element count exceed the int array bound.
            const long TargetFloatsPerChunk = 64L * 1024 * 1024;
            long perChunk = Math.Max(1, TargetFloatsPerChunk / dimension);
            _vectorsPerChunk = (int)Math.Min(perChunk, int.MaxValue / dimension);
        }

        public override bool IsReadOnly => false;

        public override void EnsureCapacity(int slotCount)
        {
            while (_capacity < slotCount)
            {
                _chunks.Add(new float[_vectorsPerChunk * _dimension]);
                _capacity += _vectorsPerChunk;
            }
        }

        public override ReadOnlySpan<float> Vector(int slot) => Slot(slot);

        public override Span<float> VectorMutable(int slot) => Slot(slot);

        private Span<float> Slot(int slot)
        {
            int chunk = slot / _vectorsPerChunk;
            int offset = (slot % _vectorsPerChunk) * _dimension;
            return _chunks[chunk].AsSpan(offset, _dimension);
        }
    }

    // Read-only storage backed by a memory-mapped file. Each slot is served as a span straight off
    // the mapped pages; the file is addressed with long offsets so the vector section can exceed the
    // .NET array length limit while each per-slot span stays within it.
    private sealed unsafe class MappedVectorBlock : VectorBlock, IDisposable
    {
        private readonly MemoryMappedFile _file;
        private readonly MemoryMappedViewAccessor _view;
        private readonly long _vectorOffset;
        private readonly int _dimension;
        private byte* _base;

        public MappedVectorBlock(MemoryMappedFile file, MemoryMappedViewAccessor view, long vectorOffset, int dimension)
        {
            _file = file;
            _view = view;
            _vectorOffset = vectorOffset;
            _dimension = dimension;
            byte* pointer = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _base = pointer + view.PointerOffset;
        }

        public override bool IsReadOnly => true;

        public override void EnsureCapacity(int slotCount)
        {
        }

        public override ReadOnlySpan<float> Vector(int slot)
            => new(_base + _vectorOffset + (long)slot * _dimension * sizeof(float), _dimension);

        public override Span<float> VectorMutable(int slot)
            => throw new NotSupportedException("Memory-mapped vectors are read-only.");

        public void Dispose()
        {
            if (_base != null)
            {
                _view.SafeMemoryMappedViewHandle.ReleasePointer();
                _base = null;
            }

            _view.Dispose();
            _file.Dispose();
        }
    }

    private sealed class Node
    {
        public Node(long id, int level)
        {
            Id = id;
            Level = level;
            Links = new List<int>[level + 1];
            for (int i = 0; i < Links.Length; i++)
            {
                Links[i] = new List<int>();
            }
        }

        public long Id { get; set; }

        public int Level { get; }

        public bool Deleted { get; set; }

        public List<int>[] Links { get; }
    }

    private sealed class Scratch
    {
        public int[] Visited = Array.Empty<int>();

        public int Version;

        public readonly PriorityQueue<Candidate, float> Candidates = new();

        public readonly PriorityQueue<Candidate, float> Nearest = new();

        public readonly List<Candidate> LayerResult = new();

        public readonly List<Candidate> PruneCandidates = new();

        public readonly List<Candidate> SelectPruned = new();

        public readonly List<int> AddSelected = new();

        public readonly List<int> PruneSelected = new();

        public int NextVersion(int nodeCount)
        {
            if (Visited.Length < nodeCount)
            {
                Array.Resize(ref Visited, Math.Max(16, nodeCount * 2));
            }

            if (++Version == 0)
            {
                Array.Clear(Visited);
                Version = 1;
            }

            return Version;
        }
    }

    private readonly record struct Candidate(int Index, float Distance);
}
