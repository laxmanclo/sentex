import numpy as np
import pytest

from sentex.ingestion.knn import build_knn, update_knn


def _random_vecs(n: int, d: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def test_build_knn_shape():
    vecs = _random_vecs(10)
    adj = build_knn(vecs, k=3)
    assert len(adj) == 10
    for neighbors in adj.values():
        assert len(neighbors) == 3


def test_build_knn_no_self_loops():
    vecs = _random_vecs(10)
    adj = build_knn(vecs, k=3)
    for i, neighbors in adj.items():
        assert all(j != i for j, _ in neighbors)


def test_build_knn_sorted_descending():
    vecs = _random_vecs(20)
    adj = build_knn(vecs, k=5)
    for neighbors in adj.values():
        sims = [s for _, s in neighbors]
        assert sims == sorted(sims, reverse=True)


def test_update_knn_extends_correctly():
    vecs_old = _random_vecs(5)
    adj = build_knn(vecs_old, k=2)

    vecs_new = np.vstack([vecs_old, _random_vecs(3, seed=99)])
    adj = update_knn(vecs_new, adj, new_start=5, k=2)

    assert len(adj) == 8
    for i in range(5, 8):
        assert i in adj
        assert all(j != i for j, _ in adj[i])


def test_build_knn_empty():
    adj = build_knn(np.empty((0, 8), dtype=np.float32), k=5)
    assert adj == {}
