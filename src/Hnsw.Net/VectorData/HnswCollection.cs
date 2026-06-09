using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
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

    private const uint SnapshotMagic = 0x44564E48; // "HNVD"
    private const int SnapshotVersion = 1;

    /// <summary>
    /// Provider-specific persistence. Writes this collection's records and the backing
    /// <see cref="HnswIndex" /> graph to <paramref name="stream" /> in a self-contained format.
    /// </summary>
    /// <remarks>
    /// This is not part of the <see cref="VectorStoreCollection{TKey, TRecord}" /> abstraction;
    /// reach it by holding the concrete <see cref="HnswCollection{TKey, TRecord}" /> type.
    /// This overload serializes records by reflection. For trimmed or NativeAOT applications,
    /// use <see cref="Save(Stream, JsonSerializerContext)" /> with a source-generated context.
    /// </remarks>
    [RequiresUnreferencedCode("Snapshot persistence serializes records by reflection and is incompatible with trimming. Use the JsonSerializerContext overload for trimming/NativeAOT.")]
    [RequiresDynamicCode("Snapshot persistence serializes records by reflection and is incompatible with NativeAOT. Use the JsonSerializerContext overload for trimming/NativeAOT.")]
    public void Save(Stream stream)
        => SaveCore(stream, SerializeByReflection, SerializeByReflection);

    /// <summary>
    /// Provider-specific persistence, AOT- and trimming-safe. Writes this collection's records and the
    /// backing <see cref="HnswIndex" /> graph to <paramref name="stream" />, serializing records with the
    /// supplied source-generated <paramref name="context" />.
    /// </summary>
    /// <param name="stream">The destination stream.</param>
    /// <param name="context">
    /// A source-generated <see cref="JsonSerializerContext" /> that provides metadata for both
    /// <typeparamref name="TKey" /> and <typeparamref name="TRecord" />.
    /// </param>
    public void Save(Stream stream, JsonSerializerContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        JsonTypeInfo keyInfo = ResolveTypeInfo(context, typeof(TKey));
        JsonTypeInfo recordInfo = ResolveTypeInfo(context, typeof(TRecord));
        SaveCore(
            stream,
            key => JsonSerializer.SerializeToUtf8Bytes(key, keyInfo),
            record => JsonSerializer.SerializeToUtf8Bytes(record, recordInfo));
    }

    /// <summary>
    /// Provider-specific persistence. Replaces this collection's contents with a snapshot
    /// previously written by <see cref="Save(Stream)" />.
    /// </summary>
    /// <remarks>
    /// This overload deserializes records by reflection. For trimmed or NativeAOT applications,
    /// use <see cref="Load(Stream, JsonSerializerContext)" /> with a source-generated context.
    /// </remarks>
    [RequiresUnreferencedCode("Snapshot persistence serializes records by reflection and is incompatible with trimming. Use the JsonSerializerContext overload for trimming/NativeAOT.")]
    [RequiresDynamicCode("Snapshot persistence serializes records by reflection and is incompatible with NativeAOT. Use the JsonSerializerContext overload for trimming/NativeAOT.")]
    public void Load(Stream stream)
        => LoadCore(stream, DeserializeByReflection<TKey>, DeserializeByReflection<TRecord>);

    /// <summary>
    /// Provider-specific persistence, AOT- and trimming-safe. Replaces this collection's contents with a
    /// snapshot previously written by either <see cref="Save(Stream)" /> overload, deserializing records
    /// with the supplied source-generated <paramref name="context" />.
    /// </summary>
    /// <param name="stream">The source stream.</param>
    /// <param name="context">
    /// A source-generated <see cref="JsonSerializerContext" /> that provides metadata for both
    /// <typeparamref name="TKey" /> and <typeparamref name="TRecord" />.
    /// </param>
    public void Load(Stream stream, JsonSerializerContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        JsonTypeInfo keyInfo = ResolveTypeInfo(context, typeof(TKey));
        JsonTypeInfo recordInfo = ResolveTypeInfo(context, typeof(TRecord));
        LoadCore(
            stream,
            json => DeserializeWithTypeInfo<TKey>(json, keyInfo),
            json => DeserializeWithTypeInfo<TRecord>(json, recordInfo));
    }

    // Shared binary framing for both the reflection and source-generated overloads. Records and keys are
    // serialized individually so the caller's context only needs metadata for TKey and TRecord.
    private void SaveCore(Stream stream, Func<TKey, byte[]> serializeKey, Func<TRecord, byte[]> serializeRecord)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite)
        {
            throw new ArgumentException("The stream is not writable.", nameof(stream));
        }

        HnswCollectionData data = GetData();
        lock (data.Lock)
        {
            // Serialize and validate every entry before writing anything, so a type mismatch or an
            // oversized payload can't leave a partially-written snapshot on the destination stream.
            var serialized = new List<(long Id, byte[] Key, byte[] Record)>(data.Records.Count);
            foreach (KeyValuePair<object, HnswCollectionData.Entry> kvp in data.Records)
            {
                if (kvp.Key is not TKey typedKey || kvp.Value.Record is not TRecord typedRecord)
                {
                    throw new InvalidOperationException(
                        $"Collection '{Name}' contains an entry whose key or record type does not match " +
                        $"the '{typeof(TKey).Name}'/'{typeof(TRecord).Name}' used to save it. The same collection " +
                        "name must always be accessed with a single key and record type.");
                }

                byte[] keyJson = serializeKey(typedKey);
                byte[] recordJson = serializeRecord(typedRecord);
                if (keyJson.Length > MaxPayloadLength || recordJson.Length > MaxPayloadLength)
                {
                    throw new InvalidOperationException(
                        $"A serialized key or record exceeds the maximum supported snapshot payload size ({MaxPayloadLength} bytes).");
                }

                serialized.Add((kvp.Value.Id, keyJson, recordJson));
            }

            using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
            {
                writer.Write(SnapshotMagic);
                writer.Write(SnapshotVersion);
                writer.Write(data.Dimension);
                writer.Write((int)_metric);
                writer.Write(data.NextId);
                writer.Write(serialized.Count);
                foreach ((long id, byte[] keyJson, byte[] recordJson) in serialized)
                {
                    writer.Write(id);
                    writer.Write(keyJson.Length);
                    writer.Write(keyJson);
                    writer.Write(recordJson.Length);
                    writer.Write(recordJson);
                }

                writer.Write(data.Index is not null);
            }

            data.Index?.Save(stream);
        }
    }

    private void LoadCore(Stream stream, Func<byte[], TKey> deserializeKey, Func<byte[], TRecord> deserializeRecord)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead)
        {
            throw new ArgumentException("The stream is not readable.", nameof(stream));
        }

        int dimension;
        DistanceMetric metric;
        long nextId;
        bool hasIndex;
        var entries = new List<(TKey Key, long Id, TRecord Record)>();
        using (var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true))
        {
            try
            {
                if (reader.ReadUInt32() != SnapshotMagic)
                {
                    throw new InvalidDataException("The stream is not an Hnsw.Net collection snapshot.");
                }

                if (reader.ReadInt32() != SnapshotVersion)
                {
                    throw new InvalidDataException("Unsupported Hnsw.Net collection snapshot version.");
                }

                dimension = reader.ReadInt32();
                metric = (DistanceMetric)reader.ReadInt32();
                nextId = reader.ReadInt64();

                if (dimension < 0)
                {
                    throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: negative vector dimension.");
                }

                if (nextId < 0)
                {
                    throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: negative next id.");
                }

                int configured = _model.VectorProperty.Dimensions;
                if (configured > 0 && dimension > 0 && dimension != configured)
                {
                    throw new InvalidDataException(
                        $"The snapshot's vector dimension ({dimension}) does not match the collection's configured dimension ({configured}).");
                }

                if (metric != _metric)
                {
                    throw new InvalidDataException(
                        $"The snapshot's distance metric ({metric}) does not match the collection's configured metric ({_metric}).");
                }

                int count = reader.ReadInt32();
                if (count < 0)
                {
                    throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: negative record count.");
                }

                for (int i = 0; i < count; i++)
                {
                    long id = reader.ReadInt64();
                    byte[] keyJson = ReadExact(reader, reader.ReadInt32());
                    byte[] recordJson = ReadExact(reader, reader.ReadInt32());
                    entries.Add((deserializeKey(keyJson), id, deserializeRecord(recordJson)));
                }

                hasIndex = reader.ReadBoolean();
            }
            catch (EndOfStreamException ex)
            {
                throw new InvalidDataException("Truncated Hnsw.Net collection snapshot.", ex);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: a key or record payload is not valid JSON.", ex);
            }
            catch (NotSupportedException ex)
            {
                throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: a key or record payload could not be deserialized.", ex);
            }
        }

        if (entries.Count > 0 && !hasIndex)
        {
            throw new InvalidDataException(
                "Corrupt Hnsw.Net collection snapshot: a non-empty collection has no index.");
        }

        if (hasIndex && dimension == 0)
        {
            throw new InvalidDataException(
                "Corrupt Hnsw.Net collection snapshot: an index is present but the vector dimension is zero.");
        }

        HnswIndex? index;
        try
        {
            index = hasIndex ? HnswIndex.Load(stream) : null;
        }
        catch (Exception ex) when (
            ex is EndOfStreamException or IOException or OverflowException
                or ArgumentException or InvalidOperationException or FormatException
            && ex is not InvalidDataException)
        {
            throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: the index is invalid.", ex);
        }

        if (index is not null && (index.Dimension != dimension || index.Metric != metric))
        {
            throw new InvalidDataException(
                $"Corrupt Hnsw.Net collection snapshot: the index header (dimension {index.Dimension}, metric {index.Metric}) " +
                $"does not match the snapshot header (dimension {dimension}, metric {metric}).");
        }

        // Build and fully validate the new state before touching the live collection, so a corrupt
        // snapshot leaves the existing contents intact rather than partially overwritten.
        var newRecords = new Dictionary<object, HnswCollectionData.Entry>();
        var newIdToKey = new Dictionary<long, TKey>();
        foreach ((TKey key, long id, TRecord record) in entries)
        {
            if (id < 0)
            {
                throw new InvalidDataException(
                    $"Corrupt Hnsw.Net collection snapshot: negative record id {id}.");
            }

            if (id >= nextId)
            {
                throw new InvalidDataException(
                    $"Corrupt Hnsw.Net collection snapshot: record id {id} is not less than the next id ({nextId}).");
            }

            ReadOnlyMemory<float> vector;
            if (index is null)
            {
                vector = ReadOnlyMemory<float>.Empty;
            }
            else if (index.TryGetVector(id, out float[] v))
            {
                vector = v;
            }
            else
            {
                throw new InvalidDataException(
                    $"Corrupt Hnsw.Net collection snapshot: the index does not contain a vector for record id {id}.");
            }

            if (!newRecords.TryAdd(key!, new HnswCollectionData.Entry { Record = record!, Id = id, Vector = vector }))
            {
                throw new InvalidDataException(
                    $"Corrupt Hnsw.Net collection snapshot: duplicate record key '{key}'.");
            }

            if (!newIdToKey.TryAdd(id, key!))
            {
                throw new InvalidDataException(
                    $"Corrupt Hnsw.Net collection snapshot: duplicate record id {id}.");
            }
        }

        // Establish/validate the record type before touching _collections so a concurrent GetCollection
        // with a different TRecord can't slip in and observe mixed-type data for this name.
        Type existingType = _collectionTypes.GetOrAdd(Name, typeof(TRecord));
        if (existingType != typeof(TRecord))
        {
            throw new InvalidOperationException(
                $"Collection '{Name}' already exists with data type '{existingType.Name}' and cannot be loaded as data type '{typeof(TRecord).Name}'.");
        }

        // Update the existing per-collection data in place so its Lock object stays stable for any
        // other threads that already captured it via GetData().
        HnswCollectionData data = _collections.GetOrAdd(Name, static _ => new HnswCollectionData());
        lock (data.Lock)
        {
            data.Records.Clear();
            data.IdToKey.Clear();
            data.Dimension = dimension;
            data.NextId = nextId;
            data.Index = index;
            foreach ((object key, HnswCollectionData.Entry entry) in newRecords)
            {
                data.Records[key] = entry;
            }

            foreach ((long id, TKey key) in newIdToKey)
            {
                data.IdToKey[id] = key!;
            }
        }
    }

    // An individual key or record payload should never be huge; cap it so a corrupt length prefix on a
    // non-seekable stream cannot trigger a denial-of-service allocation before truncation is detected.
    private const int MaxPayloadLength = 128 * 1024 * 1024;

    private static byte[] ReadExact(BinaryReader reader, int length)
    {
        if (length < 0)
        {
            throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: negative payload length.");
        }

        if (length > MaxPayloadLength)
        {
            throw new InvalidDataException("Corrupt Hnsw.Net collection snapshot: payload length exceeds the maximum supported size.");
        }

        // When the stream is seekable, the payload cannot be larger than the bytes that remain.
        Stream baseStream = reader.BaseStream;
        if (baseStream.CanSeek && length > baseStream.Length - baseStream.Position)
        {
            throw new InvalidDataException("Truncated Hnsw.Net collection snapshot.");
        }

        byte[] payload = reader.ReadBytes(length);
        if (payload.Length != length)
        {
            throw new InvalidDataException("Truncated Hnsw.Net collection snapshot.");
        }

        return payload;
    }

    private static JsonTypeInfo ResolveTypeInfo(JsonSerializerContext context, Type type)
        => context.GetTypeInfo(type) ?? throw new InvalidOperationException(
            $"The supplied JsonSerializerContext does not provide metadata for type '{type}'. " +
            $"Add a [JsonSerializable] attribute for it to the context.");

    private static T DeserializeWithTypeInfo<T>(byte[] json, JsonTypeInfo typeInfo)
        => JsonSerializer.Deserialize(json, typeInfo) is T value
            ? value
            : throw new InvalidDataException($"Invalid Hnsw.Net collection snapshot: could not deserialize a '{typeof(T)}'.");

    [RequiresUnreferencedCode("Serializes by reflection.")]
    [RequiresDynamicCode("Serializes by reflection.")]
    private static byte[] SerializeByReflection<T>(T value)
        => JsonSerializer.SerializeToUtf8Bytes(value);

    [RequiresUnreferencedCode("Deserializes by reflection.")]
    [RequiresDynamicCode("Deserializes by reflection.")]
    private static T DeserializeByReflection<T>(byte[] json)
        => JsonSerializer.Deserialize<T>(json)
            ?? throw new InvalidDataException("Invalid Hnsw.Net collection snapshot.");

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
