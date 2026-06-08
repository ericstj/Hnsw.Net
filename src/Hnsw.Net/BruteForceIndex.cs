using System.Numerics.Tensors;

namespace HnswNet;

/// <summary>
/// An exact (brute-force) nearest-neighbor index sharing <see cref="DistanceMetric" /> semantics with
/// <see cref="HnswIndex" />. It scans every vector per query, so it is intended for small collections or
/// for producing exact baselines to validate the approximate index.
/// </summary>
public sealed class BruteForceIndex
{
    private readonly Dictionary<long, int> _ids = new();
    private readonly List<long> _idBySlot = new();
    private readonly List<float[]> _vectors = new();
    private readonly List<float[]> _originalVectors = new();

    /// <summary>Initializes a new brute-force index.</summary>
    /// <param name="dimension">Vector dimension. All indexed and query vectors must have this length.</param>
    /// <param name="metric">Distance metric. Dot product uses negative inner product so lower is better.</param>
    public BruteForceIndex(int dimension, DistanceMetric metric)
    {
        if (dimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension));
        }

        Dimension = dimension;
        Metric = metric;
    }

    /// <summary>Gets the vector dimension.</summary>
    public int Dimension { get; }

    /// <summary>Gets the distance metric.</summary>
    public DistanceMetric Metric { get; }

    /// <summary>Gets the number of indexed vectors.</summary>
    public int Count => _ids.Count;

    /// <summary>Adds a vector with the specified id. Duplicate ids throw <see cref="ArgumentException" />.</summary>
    public void Add(long id, ReadOnlySpan<float> vector)
    {
        ValidateVector(vector);
        if (_ids.ContainsKey(id))
        {
            throw new ArgumentException("An item with the same id already exists.", nameof(id));
        }

        _ids.Add(id, _vectors.Count);
        _idBySlot.Add(id);
        _originalVectors.Add(vector.ToArray());
        _vectors.Add(Prepare(vector));
    }

    /// <summary>Adds a vector with the specified id.</summary>
    public void Add(long id, ReadOnlyMemory<float> vector) => Add(id, vector.Span);

    /// <summary>Removes the vector with the specified id. Returns false if the id is unknown.</summary>
    public bool Remove(long id)
    {
        if (!_ids.TryGetValue(id, out int slot))
        {
            return false;
        }

        int last = _vectors.Count - 1;
        if (slot != last)
        {
            _vectors[slot] = _vectors[last];
            _originalVectors[slot] = _originalVectors[last];
            _idBySlot[slot] = _idBySlot[last];
            _ids[_idBySlot[slot]] = slot;
        }

        _vectors.RemoveAt(last);
        _originalVectors.RemoveAt(last);
        _idBySlot.RemoveAt(last);
        _ids.Remove(id);
        return true;
    }

    /// <summary>Returns whether a vector with the specified id exists.</summary>
    public bool Contains(long id) => _ids.ContainsKey(id);

    /// <summary>Gets a copy of the original vector for an id. Returns false for unknown ids.</summary>
    public bool TryGetVector(long id, out float[] vector)
    {
        if (_ids.TryGetValue(id, out int slot))
        {
            vector = (float[])_originalVectors[slot].Clone();
            return true;
        }

        vector = Array.Empty<float>();
        return false;
    }

    /// <summary>Returns the exact nearest vectors sorted by ascending distance.</summary>
    public IReadOnlyList<(long Id, float Distance)> Search(ReadOnlySpan<float> query, int k) => Search(query, k, null);

    /// <summary>Returns the exact nearest vectors, optionally restricted to ids accepted by <paramref name="filter" />.</summary>
    public IReadOnlyList<(long Id, float Distance)> Search(ReadOnlySpan<float> query, int k, Func<long, bool>? filter)
    {
        ValidateVector(query);
        if (k <= 0 || _vectors.Count == 0)
        {
            return Array.Empty<(long Id, float Distance)>();
        }

        float[] prepared = Prepare(query);
        var matches = new List<(long Id, float Distance)>(_vectors.Count);
        for (int i = 0; i < _vectors.Count; i++)
        {
            long id = _idBySlot[i];
            if (filter is not null && !filter(id))
            {
                continue;
            }

            matches.Add((id, Distance(prepared, _vectors[i])));
        }

        matches.Sort(static (a, b) =>
        {
            int byDistance = a.Distance.CompareTo(b.Distance);
            return byDistance != 0 ? byDistance : a.Id.CompareTo(b.Id);
        });

        int resultCount = Math.Min(k, matches.Count);
        var results = new (long Id, float Distance)[resultCount];
        for (int i = 0; i < resultCount; i++)
        {
            results[i] = matches[i];
        }

        return results;
    }

    private void ValidateVector(ReadOnlySpan<float> vector)
    {
        if (vector.Length != Dimension)
        {
            throw new ArgumentException($"Expected vector dimension {Dimension}, got {vector.Length}.", nameof(vector));
        }
    }

    private float[] Prepare(ReadOnlySpan<float> vector)
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

    private float Distance(ReadOnlySpan<float> left, ReadOnlySpan<float> right) => Metric switch
    {
        DistanceMetric.Cosine => 1.0f - TensorPrimitives.Dot(left, right),
        DistanceMetric.EuclideanL2 => TensorPrimitives.Distance(left, right),
        DistanceMetric.DotProduct => -TensorPrimitives.Dot(left, right),
        _ => throw new InvalidOperationException("Unknown distance metric."),
    };
}
