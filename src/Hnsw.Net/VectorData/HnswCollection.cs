using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace HnswNet;

/// <summary>
/// A <see cref="VectorStoreCollection{TKey, TRecord}" /> backed by an in-process <see cref="HnswIndex" />.
/// Provides approximate-nearest-neighbor search over records with no external service and no native dependency.
/// </summary>
/// <typeparam name="TKey">The record key type. Any non-null type is supported.</typeparam>
/// <typeparam name="TRecord">The record type, annotated with vector-data attributes or described by a definition.</typeparam>
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix (Collection)
public class HnswCollection<TKey, TRecord> : VectorStoreCollection<TKey, TRecord>
#pragma warning restore CA1711
    where TKey : notnull
    where TRecord : class
{
    private static readonly VectorSearchOptions<TRecord> s_defaultSearchOptions = new();

    private readonly ConcurrentDictionary<string, HnswCollectionData> _collections;
    private readonly ConcurrentDictionary<string, Type> _collectionTypes;
    private readonly CollectionModel _model;
    private readonly VectorStoreCollectionMetadata _metadata;
    private readonly DistanceMetric _metric;
    private readonly string? _distanceFunction;
    private readonly int _m;
    private readonly int _efConstruction;
    private readonly int? _ef;
    private readonly int? _seed;

    /// <summary>
    /// Initializes a new instance of the <see cref="HnswCollection{TKey, TRecord}" /> class with its own storage.
    /// </summary>
    /// <param name="name">The collection name.</param>
    /// <param name="options">Optional configuration options.</param>
    [RequiresUnreferencedCode("The HNSW connector uses reflection to map records and is incompatible with trimming.")]
    [RequiresDynamicCode("The HNSW connector uses reflection to map records and is incompatible with NativeAOT.")]
    public HnswCollection(string name, HnswCollectionOptions? options = null)
        : this(internalCollections: null, internalCollectionTypes: null, name, options)
    {
    }

    [RequiresUnreferencedCode("The HNSW connector uses reflection to map records and is incompatible with trimming.")]
    [RequiresDynamicCode("The HNSW connector uses reflection to map records and is incompatible with NativeAOT.")]
    internal HnswCollection(
        ConcurrentDictionary<string, HnswCollectionData>? internalCollections,
        ConcurrentDictionary<string, Type>? internalCollectionTypes,
        string name,
        HnswCollectionOptions? options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        if (typeof(TRecord) == typeof(Dictionary<string, object?>))
        {
            throw new NotSupportedException("Dynamic records (Dictionary<string, object?>) are not supported by the HNSW connector.");
        }

        options ??= new HnswCollectionOptions();

        Name = name;
        _collections = internalCollections ?? new();
        _collectionTypes = internalCollectionTypes ?? new();
        _model = new HnswModelBuilder().Build(typeof(TRecord), typeof(TKey), options.Definition, options.EmbeddingGenerator);
        _distanceFunction = _model.VectorProperty.DistanceFunction;
        _metric = HnswDistanceMapping.ToMetric(_distanceFunction);
        _m = options.M;
        _efConstruction = options.EfConstruction;
        _ef = options.Ef;
        _seed = options.Seed;
        _metadata = new VectorStoreCollectionMetadata
        {
            VectorStoreSystemName = HnswConstants.VectorStoreSystemName,
            CollectionName = name,
        };
    }

    /// <inheritdoc />
    public override string Name { get; }

    /// <inheritdoc />
    public override Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
        => Task.FromResult(_collections.ContainsKey(Name));

    /// <inheritdoc />
    public override Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        if (_collections.TryAdd(Name, new HnswCollectionData()))
        {
            _collectionTypes.TryAdd(Name, typeof(TRecord));
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        _collections.TryRemove(Name, out _);
        _collectionTypes.TryRemove(Name, out _);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task<TRecord?> GetAsync(TKey key, RecordRetrievalOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(key);

        HnswCollectionData data = GetData();
        lock (data.Lock)
        {
            return Task.FromResult(data.Records.TryGetValue(key, out HnswCollectionData.Entry? entry)
                ? (TRecord?)entry.Record
                : default);
        }
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<TRecord> GetAsync(
        Expression<Func<TRecord, bool>> filter,
        int top,
        FilteredRecordRetrievalOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(filter);
        ArgumentOutOfRangeException.ThrowIfLessThan(top, 1);
        await Task.CompletedTask.ConfigureAwait(false);

        options ??= new FilteredRecordRetrievalOptions<TRecord>();
        Func<TRecord, bool> predicate = filter.Compile();

        List<TRecord> matches;
        HnswCollectionData data = GetData();
        lock (data.Lock)
        {
            matches = data.Records.Values
                .Select(e => (TRecord)e.Record)
                .Where(predicate)
                .Skip(options.Skip)
                .Take(top)
                .ToList();
        }

        foreach (TRecord record in matches)
        {
            yield return record;
        }
    }

    /// <inheritdoc />
    public override Task DeleteAsync(TKey key, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(key);

        HnswCollectionData data = GetData();
        lock (data.Lock)
        {
            if (data.Records.Remove(key, out HnswCollectionData.Entry? entry))
            {
                data.IdToKey.Remove(entry.Id);
                data.Index?.MarkDeleted(entry.Id);
            }
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
        => UpsertAsync([record], cancellationToken);

    /// <inheritdoc />
    public override async Task UpsertAsync(IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);

        IReadOnlyList<TRecord> recordList = records as IReadOnlyList<TRecord> ?? records.ToList();
        if (recordList.Count == 0)
        {
            return;
        }

        VectorPropertyModel vectorProperty = _model.VectorProperty;
        bool nativeVector = HnswModelBuilder.IsVectorPropertyTypeValidCore(vectorProperty.Type, out _);

        IReadOnlyList<Embedding>? generated = null;
        if (!nativeVector)
        {
            generated = await vectorProperty
                .GenerateEmbeddingsAsync(recordList.Select(r => vectorProperty.GetValueAsObject(r)), cancellationToken)
                .ConfigureAwait(false);
        }

        KeyPropertyModel keyProperty = _model.KeyProperty;
        HnswCollectionData data = GetData();
        lock (data.Lock)
        {
            for (int i = 0; i < recordList.Count; i++)
            {
                TRecord record = recordList[i];
                var key = (TKey)keyProperty.GetValueAsObject(record)!;
                if (keyProperty.IsAutoGenerated && key is Guid emptyGuid && emptyGuid == Guid.Empty)
                {
                    Guid newKey = Guid.NewGuid();
                    keyProperty.SetValue<Guid>(record, newKey);
                    key = (TKey)(object)newKey;
                }

                ReadOnlyMemory<float> vector = nativeVector
                    ? ToVector(vectorProperty.GetValueAsObject(record), vectorProperty.ModelName)
                    : ((Embedding<float>)generated![i]).Vector;

                HnswIndex index = GetOrCreateIndex(data, vector.Length);

                if (data.Records.Remove(key, out HnswCollectionData.Entry? existing))
                {
                    data.IdToKey.Remove(existing.Id);
                    index.MarkDeleted(existing.Id);
                }

                long id = data.NextId++;
                index.Add(id, vector.Span);
                data.Records[key] = new HnswCollectionData.Entry { Record = record, Id = id, Vector = vector };
                data.IdToKey[id] = key;
            }
        }
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<VectorSearchResult<TRecord>> SearchAsync<TInput>(
        TInput searchValue,
        int top,
        VectorSearchOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(searchValue);
        ArgumentOutOfRangeException.ThrowIfLessThan(top, 1);

        options ??= s_defaultSearchOptions;
        if (options.IncludeVectors && _model.EmbeddingGenerationRequired)
        {
            throw new NotSupportedException("Returning vectors is not supported when embeddings are generated by the connector.");
        }

        VectorPropertyModel vectorProperty = _model.GetVectorPropertyOrSingle(options);

        ReadOnlyMemory<float> query = searchValue switch
        {
            ReadOnlyMemory<float> m => m,
            float[] f => f,
            Embedding<float> e => e.Vector,
            _ when vectorProperty.EmbeddingGenerationDispatcher is not null
                => ((Embedding<float>)await vectorProperty.GenerateEmbeddingAsync(searchValue, cancellationToken).ConfigureAwait(false)).Vector,
            _ => throw new NotSupportedException(
                $"Search input of type '{typeof(TInput).Name}' is not supported and no embedding generator is configured. Supported vector types: {HnswModelBuilder.SupportedVectorTypes}."),
        };

        List<VectorSearchResult<TRecord>> results = Search(query, top, options);
        foreach (VectorSearchResult<TRecord> result in results)
        {
            yield return result;
        }
    }

    private List<VectorSearchResult<TRecord>> Search(ReadOnlyMemory<float> query, int top, VectorSearchOptions<TRecord> options)
    {
        var results = new List<VectorSearchResult<TRecord>>();
        HnswCollectionData data = GetData();

        lock (data.Lock)
        {
            if (data.Index is null)
            {
                return results;
            }

            Func<TRecord, bool>? predicate = options.Filter?.Compile();
            Func<long, bool>? idFilter = predicate is null
                ? null
                : id => data.IdToKey.TryGetValue(id, out object? key)
                    && data.Records.TryGetValue(key, out HnswCollectionData.Entry? entry)
                    && predicate((TRecord)entry.Record);

            int requested = top + options.Skip;
            IReadOnlyList<(long Id, float Distance)> hits = data.Index.Search(query.Span, requested, idFilter);

            bool higherIsCloser = HnswDistanceMapping.HigherScoreIsCloser(_distanceFunction);
            int skipped = 0;
            foreach ((long id, float distance) in hits)
            {
                if (!data.IdToKey.TryGetValue(id, out object? key) || !data.Records.TryGetValue(key, out HnswCollectionData.Entry? entry))
                {
                    continue;
                }

                float score = HnswDistanceMapping.ToScore(distance, _distanceFunction);
                if (options.ScoreThreshold is double threshold &&
                    (higherIsCloser ? score < threshold : score > threshold))
                {
                    continue;
                }

                if (skipped < options.Skip)
                {
                    skipped++;
                    continue;
                }

                results.Add(new VectorSearchResult<TRecord>((TRecord)entry.Record, score));
                if (results.Count >= top)
                {
                    break;
                }
            }
        }

        return results;
    }

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);

        return
            serviceKey is not null ? null :
            serviceType == typeof(VectorStoreCollectionMetadata) ? _metadata :
            serviceType.IsInstanceOfType(this) ? this :
            null;
    }

    private HnswIndex GetOrCreateIndex(HnswCollectionData data, int dimension)
    {
        if (data.Index is null)
        {
            int configured = _model.VectorProperty.Dimensions;
            int actual = configured > 0 ? configured : dimension;
            if (configured > 0 && configured != dimension)
            {
                throw new InvalidOperationException($"The vector has {dimension} dimensions but the model declares {configured}.");
            }

            var index = new HnswIndex(actual, _metric, _m, _efConstruction, _seed, allowReplaceDeleted: true);
            if (_ef is int ef)
            {
                index.Ef = ef;
            }

            data.Index = index;
            data.Dimension = actual;
        }

        return data.Index;
    }

    private HnswCollectionData GetData()
        => _collections.TryGetValue(Name, out HnswCollectionData? data)
            ? data
            : throw new VectorStoreException($"Collection '{Name}' does not exist.");

    private static ReadOnlyMemory<float> ToVector(object? value, string propertyName) => value switch
    {
        ReadOnlyMemory<float> m => m,
        float[] a => a,
        Embedding<float> e => e.Vector,
        null => throw new InvalidOperationException($"The vector property '{propertyName}' is required but was null."),
        _ => throw new InvalidOperationException($"The vector property '{propertyName}' has unsupported type '{value.GetType().Name}'."),
    };
}
