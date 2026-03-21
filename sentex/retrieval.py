"""L1 retrieval engine.

Priority-queue BFS from the highest-similarity entry point,
collecting sentences until the token budget is exhausted.
"""
from __future__ import annotations

import heapq

import numpy as np

from .knn import Adjacency
from .tokens import count_tokens


def retrieve_l1(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    sentences: list[str],
    adjacency: Adjacency,
    budget_tokens: int,
    candidate_ids: list[int] | None = None,
) -> tuple[list[str], float]:
    """Return (sentences, confidence_score).

    candidate_ids: restrict entry-point search to these indices (for per-node retrieval).
    confidence_score: cosine similarity of the best entry point.
    """
    if len(embeddings) == 0:
        return [], 0.0

    if candidate_ids is not None and len(candidate_ids) == 0:
        return [], 0.0

    # 1. Compute similarities for entry-point selection
    if candidate_ids is not None:
        idx_arr = np.array(candidate_ids, dtype=np.int32)
        subset = embeddings[idx_arr]
        sims_subset = subset @ query_vec          # (m,)
        best_local = int(np.argmax(sims_subset))
        entry = candidate_ids[best_local]
        confidence = float(sims_subset[best_local])
    else:
        sims = embeddings @ query_vec             # (n,)
        entry = int(np.argmax(sims))
        confidence = float(sims[entry])
        # Pre-compute for priority scoring
        sims_all = sims

    # 2. BFS via max-heap (negate sim for min-heap)
    visited: set[int] = set()
    collected: list[tuple[str, float]] = []
    token_total = 0

    # heap: (-similarity, sentence_idx)
    heap: list[tuple[float, int]] = [(-confidence, entry)]

    # Need full similarity vector for neighbour scoring when candidate_ids is None
    if candidate_ids is None:
        sim_lookup = sims_all
    else:
        # Build full similarity vector lazily
        sim_lookup = embeddings @ query_vec

    while heap and token_total < budget_tokens:
        neg_score, idx = heapq.heappop(heap)
        if idx in visited:
            continue
        visited.add(idx)

        sentence = sentences[idx]
        toks = count_tokens(sentence)
        if token_total + toks > budget_tokens:
            break

        collected.append((sentence, -neg_score))
        token_total += toks

        for neighbor_idx, _edge_sim in adjacency.get(idx, []):
            if neighbor_idx not in visited:
                score = float(sim_lookup[neighbor_idx])
                heapq.heappush(heap, (-score, neighbor_idx))

    # Return sentences ordered by relevance score descending
    collected.sort(key=lambda x: -x[1])
    return [s for s, _ in collected], confidence
