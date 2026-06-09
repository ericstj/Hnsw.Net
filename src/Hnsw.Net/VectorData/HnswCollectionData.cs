using System.IO.MemoryMappedFiles;

namespace HnswNet;

/// <summary>
/// Mutable storage shared by every <see cref="HnswCollection{TKey, TRecord}" /> instance that targets the same
/// collection name within a store. Holds the HNSW index, the record payloads, and the mapping between the
/// caller's keys and the <see cref="long" /> ids used by <see cref="HnswIndex" />.
/// </summary>
internal sealed class HnswCollectionData : IDisposable
{
    public readonly object Lock = new();

    public HnswIndex? Index;

    public int Dimension;

    /// <summary>Record payloads keyed by the (boxed) caller key.</summary>
    public readonly Dictionary<object, Entry> Records = new();

    /// <summary>Maps an HNSW id back to the caller key.</summary>
    public readonly Dictionary<long, object> IdToKey = new();

    public long NextId;

    /// <summary>
    /// When the data was loaded by memory-mapping a snapshot, the mapping that backs the lazily-materialized
    /// record payloads. Held for the lifetime of the data and released when it is replaced or disposed.
    /// </summary>
    public MappedRecordFile? MappedRecords;

    public void Dispose()
    {
        lock (Lock)
        {
            Index?.Dispose();
            MappedRecords?.Dispose();
            Index = null;
            MappedRecords = null;
        }
    }

    /// <summary>
    /// A record payload that is either materialized eagerly (mutation and stream-load paths) or deserialized
    /// on first access from a memory-mapped snapshot region (the mmap load path).
    /// </summary>
    public sealed class Entry
    {
        private object? _record;
        private Func<object>? _factory;

        public Entry(object record, long id, ReadOnlyMemory<float> vector)
        {
            _record = record;
            Id = id;
            Vector = vector;
        }

        public Entry(Func<object> recordFactory, long id, ReadOnlyMemory<float> vector)
        {
            _factory = recordFactory;
            Id = id;
            Vector = vector;
        }

        public long Id { get; }

        public ReadOnlyMemory<float> Vector { get; }

        /// <summary>The record, materialized on first access for mmap-backed entries.</summary>
        public object Record
        {
            get
            {
                object? record = Volatile.Read(ref _record);
                if (record is not null)
                {
                    return record;
                }

                lock (this)
                {
                    if (_record is null)
                    {
                        _record = _factory!();
                        _factory = null;
                    }

                    return _record;
                }
            }
        }
    }
}

/// <summary>
/// Owns a read-only memory mapping of a snapshot file and hands out spans over record payloads so they can be
/// deserialized lazily without copying the whole file onto the managed heap. Disposing releases the mapping;
/// spans must not be used afterward.
/// </summary>
internal sealed unsafe class MappedRecordFile : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private bool _disposed;

    public MappedRecordFile(string path)
    {
        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        try
        {
            _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _base = pointer + _view.PointerOffset;
        }
        catch
        {
            _view?.Dispose();
            _file.Dispose();
            throw;
        }
    }

    public ReadOnlySpan<byte> Slice(long offset, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<byte>(_base + offset, length);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
    }
}
