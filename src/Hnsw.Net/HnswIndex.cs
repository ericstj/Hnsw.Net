using System.Collections.Concurrent;
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
public sealed class HnswIndex
{
    private const int FormatVersion = 3;
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
    private float[] _storedVectors = Array.Empty<float>();

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
    /// Returns copies of the live ids and original vectors, suitable for rebuilding a portable index.
    /// </summary>
    public IEnumerable<(long Id, float[] Vector)> ExportItems()
    {
        _lock.EnterReadLock();
        try
        {
            var items = new List<(long, float[])>(_nodes.Count - _deletedCount);
            foreach (Node node in _nodes)
            {
                if (!node.Deleted)
                {
                    items.Add((node.Id, (float[])node.OriginalVector.Clone()));
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
        float[] originalVector = vector.ToArray();
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
                    ReplaceSlot(slot, id, originalVector, originalVector);
                    return;
                }
            }

            int level = RandomLevel();
            int newIndex = _nodes.Count;
            EnsureStoredCapacity(newIndex + 1);
            PrepareVectorInto(originalVector, StoredSpanMutable(newIndex));
            var node = new Node(id, originalVector, level);
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

    /// <summary>Gets a copy of the original vector for a live id. Returns false for unknown or deleted ids.</summary>
    public bool TryGetVector(long id, out float[] vector)
    {
        _lock.EnterReadLock();
        try
        {
            if (_ids.TryGetValue(id, out int slot) && !_nodes[slot].Deleted)
            {
                vector = (float[])_nodes[slot].OriginalVector.Clone();
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

    private void ReplaceSlot(int slot, long id, ReadOnlySpan<float> vector, float[] originalVector)
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
        node.OriginalVector = originalVector;
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
                float entryDistance = Distance(preparedQuery, StoredSpan(entryPoint));
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

            for (int n = 0; n < _nodes.Count; n++)
            {
                Node node = _nodes[n];
                writer.Write(node.Id);
                writer.Write(node.Level);
                writer.Write(node.Deleted);
                // Vectors are written as raw little-endian float blocks; on the LE platforms .NET
                // targets this is byte-identical to a per-element BinaryWriter.Write(float) loop but
                // avoids millions of scalar writes on large indexes.
                writer.Write(MemoryMarshal.AsBytes(StoredSpan(n)));
                writer.Write(MemoryMarshal.AsBytes(node.OriginalVector.AsSpan()));

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

    /// <summary>Loads an index saved by <see cref="Save" />.</summary>
    public static HnswIndex Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
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
        var index = new HnswIndex(dimension, metric, m, efConstruction, ef, entryPoint, maxLevel, allowReplaceDeleted);
        index._storedVectors = count > 0 ? new float[count * dimension] : Array.Empty<float>();

        for (int i = 0; i < count; i++)
        {
            long id = reader.ReadInt64();
            int level = reader.ReadInt32();
            bool deleted = version >= 3 && reader.ReadBoolean();
            Span<float> stored = index._storedVectors.AsSpan(i * dimension, dimension);
            ReadExactInto(reader, MemoryMarshal.AsBytes(stored));
            var originalVector = new float[dimension];
            if (version >= 2)
            {
                ReadExactInto(reader, MemoryMarshal.AsBytes(originalVector.AsSpan()));
            }
            else
            {
                stored.CopyTo(originalVector);
            }

            var node = new Node(id, originalVector, level) { Deleted = deleted };
            int layerCount = reader.ReadInt32();
            if (layerCount != level + 1)
            {
                throw new InvalidDataException("Invalid layer count in Hnsw.Net index.");
            }

            for (int layer = 0; layer < layerCount; layer++)
            {
                int linkCount = reader.ReadInt32();
                for (int link = 0; link < linkCount; link++)
                {
                    node.Links[layer].Add(reader.ReadInt32());
                }
            }

            index._ids.Add(id, i);
            index._nodes.Add(node);
            if (deleted)
            {
                index._deletedCount++;
                if (allowReplaceDeleted && i != entryPoint)
                {
                    index._deletedSlots.Push(i);
                }
            }
        }

        return index;
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

    private void EnsureStoredCapacity(int nodeCount)
    {
        int required = nodeCount * Dimension;
        if (_storedVectors.Length >= required)
        {
            return;
        }

        int capacity = _storedVectors.Length == 0 ? Dimension * 16 : _storedVectors.Length * 2;
        if (capacity < required)
        {
            capacity = required;
        }

        Array.Resize(ref _storedVectors, capacity);
    }

    private ReadOnlySpan<float> StoredSpan(int index) => _storedVectors.AsSpan(index * Dimension, Dimension);

    private Span<float> StoredSpanMutable(int index) => _storedVectors.AsSpan(index * Dimension, Dimension);

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
            foreach (int neighbor in _nodes[entryPoint].Links[layer])
            {
                float distance = Distance(query, StoredSpan(neighbor));
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

        float entryDistance = Distance(query, StoredSpan(entryPoint));
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

            foreach (int neighbor in _nodes[current.Index].Links[layer])
            {
                if (s.Visited[neighbor] == version)
                {
                    continue;
                }

                s.Visited[neighbor] = version;
                float distance = Distance(query, StoredSpan(neighbor));
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
            ReadOnlySpan<float> candidateSpan = StoredSpan(candidate.Index);
            bool good = true;
            foreach (int selected in result)
            {
                if (Distance(candidateSpan, StoredSpan(selected)) < candidate.Distance)
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

    private sealed class Node
    {
        public Node(long id, float[] originalVector, int level)
        {
            Id = id;
            OriginalVector = originalVector;
            Level = level;
            Links = new List<int>[level + 1];
            for (int i = 0; i < Links.Length; i++)
            {
                Links[i] = new List<int>();
            }
        }

        public long Id { get; set; }

        public float[] OriginalVector { get; set; }

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
