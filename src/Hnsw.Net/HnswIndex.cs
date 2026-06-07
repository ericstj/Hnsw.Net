using System.Collections.ObjectModel;
using System.Numerics.Tensors;

namespace HnswNet;

/// <summary>
/// A single-threaded HNSW approximate-nearest-neighbor index over <see cref="float" /> vectors.
/// </summary>
public sealed class HnswIndex
{
    private const int FormatVersion = 2;
    private const uint Magic = 0x31575348; // HSW1, little-endian.

    private readonly List<Node> _nodes = new();
    private readonly Dictionary<long, int> _ids = new();
    private readonly Random _random;
    private readonly double _levelMultiplier;
    private int _entryPoint = -1;
    private int _maxLevel = -1;

    private readonly HashSet<int> _visited = new();
    private readonly PriorityQueue<Candidate, float> _candidatesQueue = new();
    private readonly PriorityQueue<Candidate, float> _nearestQueue = new();
    private readonly List<Candidate> _layerResult = new();
    private readonly List<Candidate> _pruneCandidates = new();
    private readonly List<Candidate> _selectPruned = new();
    private readonly List<int> _addSelected = new();
    private readonly List<int> _pruneSelected = new();
    private readonly Comparison<Candidate> _candidateComparison;

    /// <summary>Initializes a new HNSW index.</summary>
    /// <param name="dimension">Vector dimension. All indexed and query vectors must have this length.</param>
    /// <param name="metric">Distance metric. Dot product uses negative inner product so lower is better.</param>
    /// <param name="m">Maximum number of connections per non-zero layer. Layer 0 uses <c>2 * m</c>.</param>
    /// <param name="efConstruction">Beam width used while adding vectors.</param>
    /// <param name="seed">Optional random seed for reproducible layer assignment.</param>
    public HnswIndex(int dimension, DistanceMetric metric, int m = 16, int efConstruction = 200, int? seed = null)
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
        _candidateComparison = CompareCandidates;
    }

    private HnswIndex(int dimension, DistanceMetric metric, int m, int efConstruction, int ef, int entryPoint, int maxLevel)
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

    /// <summary>Gets the number of indexed vectors.</summary>
    public int Count => _nodes.Count;

    /// <summary>
    /// Returns copies of the indexed ids and original vectors, suitable for rebuilding a portable index.
    /// </summary>
    public IEnumerable<(long Id, float[] Vector)> ExportItems()
    {
        foreach (Node node in _nodes)
        {
            yield return (node.Id, (float[])node.OriginalVector.Clone());
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
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(items);
        var index = new HnswIndex(dimension, metric, m, efConstruction, seed)
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

    /// <summary>Adds a vector with the specified id. Duplicate ids throw <see cref="ArgumentException" />.</summary>
    public void Add(long id, ReadOnlySpan<float> vector)
    {
        ValidateVector(vector);
        if (_ids.ContainsKey(id))
        {
            throw new ArgumentException("An item with the same id already exists.", nameof(id));
        }

        float[] originalVector = vector.ToArray();
        float[] storedVector = PrepareVector(originalVector);
        int level = RandomLevel();
        int newIndex = _nodes.Count;
        var node = new Node(id, storedVector, originalVector, level);
        _nodes.Add(node);
        _ids.Add(id, newIndex);

        if (_entryPoint < 0)
        {
            _entryPoint = newIndex;
            _maxLevel = level;
            return;
        }

        int entryPoint = _entryPoint;
        float entryDistance = Distance(storedVector, _nodes[entryPoint].Vector);

        for (int layer = _maxLevel; layer > level; layer--)
        {
            (entryPoint, entryDistance) = SearchGreedy(storedVector, entryPoint, entryDistance, layer);
        }

        int topLayer = Math.Min(level, _maxLevel);
        for (int layer = topLayer; layer >= 0; layer--)
        {
            SearchLayer(storedVector, entryPoint, EfConstruction, layer);
            SelectNeighbors(storedVector, _layerResult, MaxConnections(layer), _addSelected);
            foreach (int neighbor in _addSelected)
            {
                node.Links[layer].Add(neighbor);
                List<int> links = _nodes[neighbor].Links[layer];
                if (!links.Contains(newIndex))
                {
                    links.Add(newIndex);
                }
                PruneConnections(neighbor, layer);
            }

            if (_layerResult.Count > 0)
            {
                Candidate nearest = _layerResult[0];
                entryPoint = nearest.Index;
                entryDistance = nearest.Distance;
            }
        }

        if (level > _maxLevel)
        {
            _entryPoint = newIndex;
            _maxLevel = level;
        }
    }

    /// <summary>Adds a vector with the specified id.</summary>
    public void Add(long id, ReadOnlyMemory<float> vector) => Add(id, vector.Span);

    /// <summary>
    /// Searches for the nearest vectors. Results are sorted by ascending distance; for dot product this means most similar first.
    /// </summary>
    public IReadOnlyList<(long Id, float Distance)> Search(ReadOnlySpan<float> query, int k)
    {
        ValidateVector(query);
        if (k <= 0 || _entryPoint < 0)
        {
            return Array.Empty<(long Id, float Distance)>();
        }

        float[] preparedQuery = PrepareVector(query);
        int entryPoint = _entryPoint;
        float entryDistance = Distance(preparedQuery, _nodes[entryPoint].Vector);
        for (int layer = _maxLevel; layer > 0; layer--)
        {
            (entryPoint, entryDistance) = SearchGreedy(preparedQuery, entryPoint, entryDistance, layer);
        }

        int ef = Math.Max(Ef, k);
        SearchLayer(preparedQuery, entryPoint, ef, 0);
        int resultCount = Math.Min(k, _layerResult.Count);
        var results = new (long Id, float Distance)[resultCount];
        for (int i = 0; i < resultCount; i++)
        {
            Candidate candidate = _layerResult[i];
            results[i] = (_nodes[candidate.Index].Id, candidate.Distance);
        }

        return results;
    }

    /// <summary>Saves the index to a stream using a versioned binary format.</summary>
    public void Save(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
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

        foreach (Node node in _nodes)
        {
            writer.Write(node.Id);
            writer.Write(node.Level);
            for (int i = 0; i < Dimension; i++)
            {
                writer.Write(node.Vector[i]);
            }
            for (int i = 0; i < Dimension; i++)
            {
                writer.Write(node.OriginalVector[i]);
            }

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
        var index = new HnswIndex(dimension, metric, m, efConstruction, ef, entryPoint, maxLevel);

        for (int i = 0; i < count; i++)
        {
            long id = reader.ReadInt64();
            int level = reader.ReadInt32();
            var vector = new float[dimension];
            for (int j = 0; j < dimension; j++)
            {
                vector[j] = reader.ReadSingle();
            }
            float[] originalVector;
            if (version >= 2)
            {
                originalVector = new float[dimension];
                for (int j = 0; j < dimension; j++)
                {
                    originalVector[j] = reader.ReadSingle();
                }
            }
            else
            {
                originalVector = (float[])vector.Clone();
            }

            var node = new Node(id, vector, originalVector, level);
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

    private int RandomLevel()
    {
        double sample = Math.Max(_random.NextDouble(), double.Epsilon);
        return (int)(-Math.Log(sample) * _levelMultiplier);
    }

    private (int Index, float Distance) SearchGreedy(float[] query, int entryPoint, float entryDistance, int layer)
    {
        bool changed;
        do
        {
            changed = false;
            foreach (int neighbor in _nodes[entryPoint].Links[layer])
            {
                float distance = Distance(query, _nodes[neighbor].Vector);
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

    private void SearchLayer(float[] query, int entryPoint, int ef, int layer)
    {
        _visited.Clear();
        _candidatesQueue.Clear();
        _nearestQueue.Clear();
        _visited.Add(entryPoint);

        float entryDistance = Distance(query, _nodes[entryPoint].Vector);
        var entry = new Candidate(entryPoint, entryDistance);
        _candidatesQueue.Enqueue(entry, entryDistance);
        _nearestQueue.Enqueue(entry, -entryDistance);
        float lowerBound = entryDistance;

        while (_candidatesQueue.Count > 0)
        {
            Candidate current = _candidatesQueue.Dequeue();
            if (current.Distance > lowerBound)
            {
                break;
            }

            foreach (int neighbor in _nodes[current.Index].Links[layer])
            {
                if (!_visited.Add(neighbor))
                {
                    continue;
                }

                float distance = Distance(query, _nodes[neighbor].Vector);
                if (_nearestQueue.Count < ef || distance < lowerBound)
                {
                    var candidate = new Candidate(neighbor, distance);
                    _candidatesQueue.Enqueue(candidate, distance);
                    _nearestQueue.Enqueue(candidate, -distance);
                    if (_nearestQueue.Count > ef)
                    {
                        _nearestQueue.Dequeue();
                    }

                    _nearestQueue.TryPeek(out _, out float priority);
                    lowerBound = -priority;
                }
            }
        }

        _layerResult.Clear();
        foreach ((Candidate element, float _) in _nearestQueue.UnorderedItems)
        {
            _layerResult.Add(element);
        }

        _layerResult.Sort(_candidateComparison);
    }

    private void SelectNeighbors(float[] query, List<Candidate> candidates, int maxConnections, List<int> result)
    {
        result.Clear();
        _selectPruned.Clear();
        candidates.Sort(_candidateComparison);
        foreach (Candidate candidate in candidates)
        {
            bool good = true;
            foreach (int selected in result)
            {
                if (Distance(_nodes[candidate.Index].Vector, _nodes[selected].Vector) < candidate.Distance)
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
                _selectPruned.Add(candidate);
            }
        }

        foreach (Candidate candidate in _selectPruned)
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

    private void PruneConnections(int nodeIndex, int layer)
    {
        List<int> links = _nodes[nodeIndex].Links[layer];
        int maxConnections = MaxConnections(layer);
        if (links.Count <= maxConnections)
        {
            return;
        }

        float[] vector = _nodes[nodeIndex].Vector;
        _pruneCandidates.Clear();
        foreach (int neighbor in links)
        {
            _pruneCandidates.Add(new Candidate(neighbor, Distance(vector, _nodes[neighbor].Vector)));
        }

        SelectNeighbors(vector, _pruneCandidates, maxConnections, _pruneSelected);
        links.Clear();
        links.AddRange(_pruneSelected);
    }

    private int CompareCandidates(Candidate left, Candidate right)
    {
        int byDistance = left.Distance.CompareTo(right.Distance);
        return byDistance != 0 ? byDistance : _nodes[left.Index].Id.CompareTo(_nodes[right.Index].Id);
    }

    private int MaxConnections(int layer) => layer == 0 ? 2 * M : M;

    private float Distance(ReadOnlySpan<float> left, ReadOnlySpan<float> right) => Metric switch
    {
        DistanceMetric.Cosine => 1.0f - TensorPrimitives.Dot(left, right),
        DistanceMetric.EuclideanL2 => TensorPrimitives.Distance(left, right),
        DistanceMetric.DotProduct => -TensorPrimitives.Dot(left, right),
        _ => throw new InvalidOperationException("Unknown distance metric."),
    };

    private sealed class Node
    {
        public Node(long id, float[] vector, float[] originalVector, int level)
        {
            Id = id;
            Vector = vector;
            OriginalVector = originalVector;
            Level = level;
            Links = new List<int>[level + 1];
            for (int i = 0; i < Links.Length; i++)
            {
                Links[i] = new List<int>();
            }
        }

        public long Id { get; }

        public float[] Vector { get; }

        public float[] OriginalVector { get; }

        public int Level { get; }

        public List<int>[] Links { get; }
    }

    private readonly record struct Candidate(int Index, float Distance);
}
