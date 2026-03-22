"""KNN adjacency builder.

Maintains an adjacency dict:
    adjacency[i] = [(j, cosine_similarity), ...]  # K nearest neighbors of sentence i

Uses the full embedding matrix for brute-force cosine search.
Swappable for approximate search (faiss, hnswlib) above ~10k sentences.
"""
from __future__ import annotations

import numpy as np


Adjacency = dict[int, list[tuple[int, float]]]


def build_knn(embeddings: np.ndarray, k: int = 5) -> Adjacency:
    """Build KNN adjacency from scratch.  O(n²) — fine up to ~10k sentences."""
    n = len(embeddings)
    if n == 0:
        return {}

    # embeddings are already L2-normalised → dot product == cosine similarity
    sim_matrix = embeddings @ embeddings.T  # (n, n)
    np.fill_diagonal(sim_matrix, -1.0)      # exclude self

    adjacency: Adjacency = {}
    k_actual = min(k, n - 1)
    for i in range(n):
        top_k = np.argpartition(sim_matrix[i], -k_actual)[-k_actual:]
        neighbors = sorted(
            [(int(j), float(sim_matrix[i, j])) for j in top_k],
            key=lambda x: -x[1],
        )
        adjacency[i] = neighbors

    return adjacency


def update_knn(
    embeddings: np.ndarray,
    adjacency: Adjacency,
    new_start: int,
    k: int = 5,
) -> Adjacency:
    """Incrementally update adjacency for sentences [new_start:].

    Also patches existing sentences' neighbor lists to include new candidates.
    """
    n = len(embeddings)
    if new_start >= n:
        return adjacency

    new_vecs = embeddings[new_start:]          # (m, D)
    all_sims = new_vecs @ embeddings.T          # (m, n)

    for rel_i, i in enumerate(range(new_start, n)):
        sims = all_sims[rel_i].copy()
        sims[i] = -1.0                          # exclude self
        k_actual = min(k, n - 1)
        top_k = np.argpartition(sims, -k_actual)[-k_actual:]
        adjacency[i] = sorted(
            [(int(j), float(sims[j])) for j in top_k],
            key=lambda x: -x[1],
        )

    # Patch existing nodes: check if any new sentence should be a neighbor
    if new_start > 0:
        old_vecs = embeddings[:new_start]       # (old, D)
        cross_sims = old_vecs @ new_vecs.T      # (old, m)

        for old_i in range(new_start):
            existing = adjacency.get(old_i, [])
            min_sim = existing[-1][1] if len(existing) == k else -1.0
            candidates = [
                (new_start + rel_j, float(cross_sims[old_i, rel_j]))
                for rel_j in range(len(new_vecs))
                if float(cross_sims[old_i, rel_j]) > min_sim
            ]
            if candidates:
                existing = existing + candidates
                existing.sort(key=lambda x: -x[1])
                adjacency[old_i] = existing[:k]

    return adjacency
