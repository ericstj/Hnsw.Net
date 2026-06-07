# Third-Party Notices

This project, **Hnsw.Net**, is a pure managed implementation of the HNSW
approximate-nearest-neighbor algorithm.

## HNSW paper

- Paper: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- Authors: Yu. A. Malkov and D. A. Yashunin
- arXiv: https://arxiv.org/abs/1603.09320

The implementation follows the algorithm described in the paper, including the
multi-layer graph, greedy upper-layer search, beam search during construction,
and the neighbor-selection heuristic.

## hnswlib

- Project: https://github.com/nmslib/hnswlib
- License: Apache License 2.0

hnswlib is credited as a widely used open-source implementation and reference
for HNSW behavior and terminology. The test suite also uses hnswlib to generate
a committed behavioral parity oracle. Hnsw.Net is implemented from the
published algorithm and does not copy hnswlib source code.
