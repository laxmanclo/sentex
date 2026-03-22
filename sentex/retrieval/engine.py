"""L1 retrieval engine.

Priority-queue BFS from the highest-similarity entry point,
collecting sentences until the token budget is exhausted.

Convergence-based early exit: if the top-k result set stabilises for
`convergence_patience` consecutive heap pops, BFS terminates early.
This cuts retrieval time significantly on large graphs (10k+ sentences)
where the relevant cluster is tight and surrounded by noise.
"""
from __future__ import annotations

import heapq

import numpy as np

from ..ingestion.knn import Adjacency
from ..core.tokens import count_tokens


def retrieve_l1(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    sentences: list[str],
    adjacency: Adjacency,
    budget_tokens: int,
    candidate_ids: list[int] | None = None,
    scope_ids: set[int] | None = None,
    convergence_k: int = 5,
    convergence_patience: int = 3,
) -> tuple[list[str], float, bool]:
    """Return (sentences, confidence_score, converged).

    Args:
        candidate_ids:  Restrict entry-point search to these indices.
        scope_ids:      If set, BFS traversal is restricted to these sentence
                        indices. Prevents cross-node contamination when retrieving
                        from a specific node — without this, BFS follows cross-node
                        KNN edges and mixes sentences from other agents.
                        Defaults to None (unrestricted) for global retrieval.
        convergence_k:         Track top-k sentences for convergence check.
        convergence_patience:  Stop if top-k set is unchanged for this many
                               consecutive heap pops.

    Returns:
        sentences:   Collected sentences sorted by relevance descending.
        confidence:  Cosine similarity of the best entry point.
        converged:   True if BFS terminated via convergence (not budget exhaustion).
    """
    if len(embeddings) == 0:
        return [], 0.0, False

    if candidate_ids is not None and len(candidate_ids) == 0:
        return [], 0.0, False

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

    # Full similarity vector for neighbour scoring
    sim_lookup = embeddings @ query_vec if candidate_ids is None else embeddings @ query_vec

    # 2. BFS via max-heap (negate sim for min-heap)
    visited: set[int] = set()
    collected: list[tuple[str, float]] = []
    collected_idxs: list[int] = []   # parallel to collected — stores global sentence idx
    token_total = 0

    # heap: (-similarity, sentence_idx)
    heap: list[tuple[float, int]] = [(-confidence, entry)]

    # Convergence tracking: top-k set of collected sentence indices
    top_k_set: frozenset[int] = frozenset()
    stable_streak = 0
    converged = False

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
        collected_idxs.append(idx)
        token_total += toks

        # Convergence check: compare current top-k *global sentence indices* to previous.
        # Must NOT use positional indices into collected — those change meaning as
        # collected grows, making any two lists of the same length look "converged".
        current_top_k = frozenset(
            gidx for gidx, _ in sorted(
                zip(collected_idxs, [s for _, s in collected]),
                key=lambda x: -x[1]
            )[:convergence_k]
        )

        if len(collected) >= convergence_k:
            if current_top_k == top_k_set:
                stable_streak += 1
                if stable_streak >= convergence_patience:
                    converged = True
                    break
            else:
                stable_streak = 0
            top_k_set = current_top_k

        # Expand neighbours — respect scope_ids to prevent cross-node contamination
        for neighbor_idx, _edge_sim in adjacency.get(idx, []):
            if neighbor_idx not in visited:
                if scope_ids is not None and neighbor_idx not in scope_ids:
                    continue  # skip sentences outside this node's scope
                score = float(sim_lookup[neighbor_idx])
                heapq.heappush(heap, (-score, neighbor_idx))

    # Return sentences ordered by relevance score descending
    collected.sort(key=lambda x: -x[1])
    return [s for s, _ in collected], confidence, converged
