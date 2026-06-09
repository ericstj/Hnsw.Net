using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;

namespace HnswNet;

/// <summary>
/// An in-process <see cref="VectorStore" /> that stores records in <see cref="HnswIndex" /> instances.
/// Provides approximate-nearest-neighbor search with no external service and no native dependency.
/// </summary>
public sealed class HnswVectorStore : VectorStore
{
    private readonly ConcurrentDictionary<string, HnswCollectionData> _collections = new();
    private readonly ConcurrentDictionary<string, (Type Key, Type Record)> _collectionTypes = new();
    private readonly VectorStoreMetadata _metadata;
    private readonly IEmbeddingGenerator? _embeddingGenerator;
    private readonly int _m;
    private readonly int _efConstruction;
    private readonly int? _ef;
    private readonly int? _seed;

    /// <summary>
    /// Initializes a new instance of the <see cref="HnswVectorStore" /> class.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    public HnswVectorStore(HnswVectorStoreOptions? options = null)
    {
        options ??= new HnswVectorStoreOptions();
        _embeddingGenerator = options.EmbeddingGenerator;
        _m = options.M;
        _efConstruction = options.EfConstruction;
        _ef = options.Ef;
        _seed = options.Seed;
        _metadata = new VectorStoreMetadata { VectorStoreSystemName = HnswConstants.VectorStoreSystemName };
    }

    /// <inheritdoc />
    [RequiresUnreferencedCode("The HNSW connector uses reflection to map records and is incompatible with trimming.")]
    [RequiresDynamicCode("The HNSW connector uses reflection to map records and is incompatible with NativeAOT.")]
    public override HnswCollection<TKey, TRecord> GetCollection<TKey, TRecord>(string name, VectorStoreCollectionDefinition? definition = null)
    {
        if (typeof(TRecord) == typeof(Dictionary<string, object?>))
        {
            throw new ArgumentException("Dynamic records (Dictionary<string, object?>) are not supported by the HNSW connector. Use a strongly-typed record.");
        }

        if (_collectionTypes.TryGetValue(name, out (Type Key, Type Record) existing)
            && (existing.Record != typeof(TRecord) || existing.Key != typeof(TKey)))
        {
            throw new InvalidOperationException($"Collection '{name}' already exists with key/data type '{existing.Key.Name}'/'{existing.Record.Name}' and cannot be re-created with key/data type '{typeof(TKey).Name}'/'{typeof(TRecord).Name}'.");
        }

        return new HnswCollection<TKey, TRecord>(
            _collections,
            _collectionTypes,
            name,
            new HnswCollectionOptions
            {
                Definition = definition,
                EmbeddingGenerator = _embeddingGenerator,
                M = _m,
                EfConstruction = _efConstruction,
                Ef = _ef,
                Seed = _seed,
            });
    }

    /// <inheritdoc />
    [RequiresUnreferencedCode("The HNSW connector uses reflection to map records and is incompatible with trimming.")]
    [RequiresDynamicCode("The HNSW connector uses reflection to map records and is incompatible with NativeAOT.")]
    public override VectorStoreCollection<object, Dictionary<string, object?>> GetDynamicCollection(string name, VectorStoreCollectionDefinition definition)
        => throw new NotSupportedException("Dynamic collections are not supported by the HNSW connector. Use a strongly-typed record with GetCollection<TKey, TRecord>.");

    /// <inheritdoc />
    public override async IAsyncEnumerable<string> ListCollectionNamesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        foreach (string name in _collections.Keys)
        {
            yield return name;
        }
    }

    /// <inheritdoc />
    public override Task<bool> CollectionExistsAsync(string name, CancellationToken cancellationToken = default)
        => Task.FromResult(_collections.ContainsKey(name));

    /// <inheritdoc />
    public override Task EnsureCollectionDeletedAsync(string name, CancellationToken cancellationToken = default)
    {
        if (_collections.TryRemove(name, out HnswCollectionData? data))
        {
            data.Dispose();
        }

        _collectionTypes.TryRemove(name, out _);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (HnswCollectionData data in _collections.Values)
            {
                data.Dispose();
            }
        }

        base.Dispose(disposing);
    }

    /// <inheritdoc />
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);

        return
            serviceKey is not null ? null :
            serviceType == typeof(VectorStoreMetadata) ? _metadata :
            serviceType.IsInstanceOfType(this) ? this :
            null;
    }
}
