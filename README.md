# Hnsw.Net

A pure C# (`net10.0`), **no native dependency** implementation of HNSW
(Hierarchical Navigable Small World graphs) for approximate nearest-neighbor
search over `float` vectors.

Implemented from the published HNSW algorithm
([Malkov & Yashunin](https://arxiv.org/abs/1603.09320)); behavior is validated
against the reference [hnswlib](https://github.com/nmslib/hnswlib). See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for full attribution.

## Features

- Managed implementation of HNSW with multi-layer graphs and the paper's
  neighbor-selection heuristic.
- Supports cosine, Euclidean L2, and dot-product similarity.
- SIMD-accelerated distance calculations via `System.Numerics.Tensors`
  (`TensorPrimitives`).
- Thread-safe concurrent search; builds and modifications are serialized.
- Predicate filtering: restrict a search to ids accepted by a caller-supplied
  `Func<long, bool>`.
- Soft delete (`MarkDeleted`/`UnmarkDeleted`) with optional reuse of freed slots
  on subsequent `Add` (`allowReplaceDeleted`).
- Exact `BruteForceIndex` companion for small collections and for producing
  exact baselines.
- Versioned binary save/load round-trips indexes (including deleted state)
  without native dependencies.
- Portable export/rebuild bridge for stored ids and vectors.
- In-box `Microsoft.Extensions.VectorData` connector (`HnswVectorStore` /
  `HnswCollection<TKey, TRecord>`) for the standard .NET vector-store abstractions,
  with optional `Microsoft.Extensions.AI` embedding generation.
- Behavioral parity validation against a committed Python hnswlib oracle.

## Usage

```csharp
using HnswNet;

var index = new HnswIndex(dimension: 3, DistanceMetric.Cosine, seed: 42)
{
    Ef = 100,
};

index.Add(1, [1, 0, 0]);
index.Add(2, [0, 1, 0]);

foreach ((long id, float distance) in index.Search([0.9f, 0.1f, 0], k: 1))
{
    Console.WriteLine($"{id}: {distance}");
}
```

Results are ordered by ascending distance. For dot product, the distance is the
negative inner product, so the most similar vectors are returned first.

Duplicate ids throw `ArgumentException`. Cosine vectors are normalized when added
and queries are normalized during search.

## Concurrency, filtering, and deletion

Searches are thread-safe and may run concurrently; concurrent results are
identical to serial ones. Builds and modifications (`Add`, `MarkDeleted`,
`UnmarkDeleted`) are serialized under a single writer.

```csharp
// Restrict results to ids your application currently considers valid.
var results = index.Search(query, k: 10, filter: id => allowedIds.Contains(id));

// Soft delete keeps the vector in the graph for connectivity but excludes it
// from results; it can be restored later.
index.MarkDeleted(2);
index.UnmarkDeleted(2);

// Opt into slot reuse so deletions do not grow the backing store unbounded.
var churningIndex = new HnswIndex(dim, DistanceMetric.Cosine, allowReplaceDeleted: true);
```

`BruteForceIndex` is an exact companion that scans every vector per query. Use it
for small collections or to validate the approximate index.

## Scope

This is a full port of the hnswlib runtime feature set (build, search, filtering,
concurrent search, soft delete and slot reuse, brute force, persistence). Two
hnswlib capabilities are intentionally excluded:

- **Fine-grained parallel build.** Hnsw.Net serializes graph mutation under a
  single writer. Concurrent search is supported, but build is single-writer; the
  lock-free per-element link locking hnswlib uses for multithreaded insertion is
  out of scope (high complexity, low value for the target workloads).
- **hnswlib native index format interop.** hnswlib's on-disk layout is not a
  stable, portable wire format. Hnsw.Net uses its own versioned format and a
  portable export/rebuild bridge instead (see below).

## Portable export/rebuild

`Save`/`Load` is the canonical Hnsw.Net index format. hnswlib's native index
format is intentionally not used for interop because it is not a stable,
portable wire format. For cross-machine or cross-version portability, export
ids and vectors and rebuild:

```csharp
IEnumerable<(long Id, float[] Vector)> items = index.ExportItems();

HnswIndex rebuilt = HnswIndex.Build(
    dimension,
    DistanceMetric.Cosine,
    items,
    m: 16,
    efConstruction: 200,
    ef: 100,
    seed: 42);
```

`ExportItems` returns copies of the original vectors passed to `Add`. Cosine
vectors are still stored normalized internally for search, but the portable
export preserves the original input values; rebuilding normalizes them again.

## hnswlib parity validation

`tests/Hnsw.Net.Tests/gen_parity.py` generates the committed
`parity_vectors.bin` and `oracle_parity.json` files with Python `hnswlib` and
NumPy. The C# parity tests load those exact vectors and assert that Hnsw.Net
recall@10 is at least 95%, within three percentage points of hnswlib recall,
and has at least 90% mean neighbor overlap with hnswlib for cosine, Euclidean
L2, and dot product. The current oracle uses 2,000 vectors, 50 queries,
dimension 64, `M=24`, `efConstruction=300`, and `ef=220`; the committed
oracle currently reaches 100% Hnsw.Net recall, 100% hnswlib recall, and 100%
Hnsw.Net/hnswlib neighbor overlap for all three metrics.

## Layout

| Path | Description |
| --- | --- |
| `src/Hnsw.Net/` | The library. |
| `tests/Hnsw.Net.Tests/` | xUnit correctness and recall tests. |
| `bench/Hnsw.Net.Benchmarks/` | BenchmarkDotNet build, search, and persistence benchmarks. |

## Building & testing

```pwsh
dotnet build -c Release
dotnet test  -c Release
```

Run benchmarks with:

```pwsh
dotnet run -c Release --project bench\Hnsw.Net.Benchmarks\Hnsw.Net.Benchmarks.csproj
```

## License

MIT — see [LICENSE](LICENSE).
