# Hnsw.Net

A pure C# (`net10.0`), **dependency-free** implementation of HNSW (Hierarchical
Navigable Small World graphs) for approximate nearest-neighbor search over
`float` vectors. No native binaries, no P/Invoke.

> Implemented from the published HNSW algorithm. Not affiliated with or derived
> from the source of any other HNSW library. See the project repository for full
> attribution and third-party notices.

## Features

- Managed multi-layer HNSW with the paper's neighbor-selection heuristic.
- Cosine, Euclidean L2, and dot-product similarity.
- SIMD-accelerated distance kernels (`System.Numerics.Tensors` + a custom vectorized dot product).
- Thread-safe concurrent search (builds and modifications are serialized).
- Predicate filtering, soft delete (`MarkDeleted`/`UnmarkDeleted`) with optional slot reuse.
- Exact `BruteForceIndex` companion for small collections and exact baselines.
- Versioned binary save/load (including deleted state), plus a portable export/rebuild bridge.
- Behavior validated against a Python `hnswlib` parity oracle.

## Installation

```pwsh
dotnet add package Hnsw.Net
```

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

// Filter results, and soft-delete without rebuilding.
var some = index.Search([0.9f, 0.1f, 0], k: 1, filter: id => id != 2);
index.MarkDeleted(1);
```

Results are ordered by ascending distance. For dot product the distance is the
negative inner product, so the most similar vectors are returned first. Cosine
vectors are normalized on add and on query. Duplicate ids throw `ArgumentException`.
Searches are thread-safe and may run concurrently; build and modification are
single-writer.

## Scope

This is a full port of the hnswlib runtime feature set (build, search, filtering,
concurrent search, soft delete and slot reuse, brute force, persistence). Two
hnswlib capabilities are intentionally excluded: fine-grained parallel build
(Hnsw.Net builds under a single writer) and hnswlib native index-format interop
(Hnsw.Net uses its own versioned format plus a portable export/rebuild bridge).

## Persistence and portability

`Save`/`Load` round-trips an index in Hnsw.Net's own versioned binary format. For
cross-machine or cross-version portability, export ids and vectors and rebuild:

```csharp
IEnumerable<(long Id, float[] Vector)> items = index.ExportItems();

HnswIndex rebuilt = HnswIndex.Build(
    dimension: 64,
    DistanceMetric.Cosine,
    items,
    m: 16,
    efConstruction: 200,
    ef: 100,
    seed: 42);
```

## Upstream / attribution

- HNSW paper — Malkov & Yashunin: https://arxiv.org/abs/1603.09320
- hnswlib (reference implementation, parity oracle): https://github.com/nmslib/hnswlib

## License

MIT. See the project repository for the license and third-party notices:
https://github.com/ericstj/Hnsw.Net
