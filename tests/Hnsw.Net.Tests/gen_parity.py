import json
import struct
from pathlib import Path

import hnswlib
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_FILE = "parity_vectors.bin"
ORACLE_FILE = "oracle_parity.json"

SEED = 20260607
N = 2000
D = 64
Q = 50
K = 10
M = 24
EF_CONSTRUCTION = 300
EF = 220
HNSWLIB_SEED = 42


def exact_ids(vectors: np.ndarray, query: np.ndarray, metric: str) -> list[int]:
    if metric == "Cosine":
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query)
        distances = 1.0 - (vectors @ query) / (vector_norms * query_norm)
    elif metric == "EuclideanL2":
        diff = vectors - query
        distances = np.einsum("ij,ij->i", diff, diff)
    elif metric == "DotProduct":
        distances = -(vectors @ query)
    else:
        raise ValueError(metric)

    return np.lexsort((np.arange(len(distances)), distances))[:K].astype(int).tolist()


def hnswlib_ids(vectors: np.ndarray, queries: np.ndarray, space: str) -> list[list[int]]:
    index = hnswlib.Index(space=space, dim=D)
    index.init_index(max_elements=N, ef_construction=EF_CONSTRUCTION, M=M, random_seed=HNSWLIB_SEED)
    index.add_items(vectors, np.arange(N))
    index.set_ef(EF)
    labels, _ = index.knn_query(queries, k=K)
    return labels.astype(int).tolist()


def main() -> None:
    rng = np.random.default_rng(SEED)
    vectors = rng.uniform(-1.0, 1.0, size=(N, D)).astype(np.float32)
    queries = rng.uniform(-1.0, 1.0, size=(Q, D)).astype(np.float32)

    with (ROOT / DATA_FILE).open("wb") as f:
        f.write(b"HNPV1\0")
        f.write(struct.pack("<iii", D, N, Q))
        f.write(vectors.tobytes(order="C"))
        f.write(queries.tobytes(order="C"))

    spaces = {
        "Cosine": "cosine",
        "EuclideanL2": "l2",
        "DotProduct": "ip",
    }
    metrics = {}
    for metric, space in spaces.items():
        hnsw = hnswlib_ids(vectors, queries, space)
        exact = [exact_ids(vectors, query, metric) for query in queries]
        metrics[metric] = {
            "hnswlibIds": hnsw,
            "exactIds": exact,
        }

    oracle = {
        "dataFile": DATA_FILE,
        "seed": SEED,
        "dimension": D,
        "vectorCount": N,
        "queryCount": Q,
        "k": K,
        "parameters": {
            "m": M,
            "efConstruction": EF_CONSTRUCTION,
            "ef": EF,
            "hnswlibSeed": HNSWLIB_SEED,
        },
        "distanceNotes": {
            "Cosine": "hnswlib and Hnsw.Net both rank by 1 - cosine similarity.",
            "EuclideanL2": "hnswlib reports squared L2 distance; neighbor ids match L2 ranking.",
            "DotProduct": "hnswlib ranks inner product as 1 - dot; Hnsw.Net uses -dot. Neighbor ids are comparable.",
        },
        "metrics": metrics,
    }
    (ROOT / ORACLE_FILE).write_text(json.dumps(oracle, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
