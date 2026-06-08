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
- Versioned binary save/load, plus a portable export/rebuild bridge.
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
```

Results are ordered by ascending distance. For dot product the distance is the
negative inner product, so the most similar vectors are returned first. Cosine
vectors are normalized on add and on query. Duplicate ids throw `ArgumentException`.
`HnswIndex` is single-threaded for build and search.

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
