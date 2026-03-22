"""Retrieval quality evaluation framework.

Measures how well Sentex retrieves relevant content compared to ground truth.
Metrics: Precision@k, Recall@k, MRR, NDCG@k, token efficiency.

Usage:
    from sentex import ContextGraph
    from sentex.eval import RetrievalEvaluator, EvalCase

    graph = ContextGraph()
    # ... ingest content ...

    evaluator = RetrievalEvaluator(graph)
    result = evaluator.evaluate([
        EvalCase(
            query="how do T-cells work",
            relevant_node_ids=["resources/research"],
        ),
    ])
    print(result.summary())

Building a dataset from production usage (no manual labeling):
    from sentex.eval import dataset_from_store
    from sentex import MemoryStore

    store = MemoryStore("sentex.db")
    cases = dataset_from_store(store, min_hits=2)
    result = evaluator.evaluate(cases)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.graph import ContextGraph
    from ..storage.store import MemoryStore


# ------------------------------------------------------------------
# Dataset types
# ------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single evaluation query with ground-truth relevant nodes."""
    query: str
    relevant_node_ids: list[str]       # which nodes should be returned
    relevant_sentence_hints: list[str] = field(default_factory=list)  # optional


@dataclass
class PerQueryResult:
    query: str
    retrieved_node_ids: list[str]      # top-k from scan_nodes
    relevant_node_ids: list[str]
    hits: list[str]                    # intersection
    reciprocal_rank: float
    precision: float
    recall: float
    ndcg: float
    mean_confidence: float
    tokens_delivered: int


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    num_cases: int
    precision_at_k: dict[int, float]   # k → mean precision
    recall_at_k: dict[int, float]
    mrr: float                         # Mean Reciprocal Rank
    ndcg_at_k: dict[int, float]
    mean_confidence: float
    mean_tokens_delivered: float
    layer_distribution: dict[str, float]   # fraction served at each layer
    per_query: list[PerQueryResult]

    def summary(self) -> str:
        lines = [
            f"Eval: {self.num_cases} queries",
            f"  MRR:          {self.mrr:.3f}",
        ]
        for k, v in sorted(self.precision_at_k.items()):
            lines.append(f"  P@{k:<3}         {v:.3f}")
        for k, v in sorted(self.recall_at_k.items()):
            lines.append(f"  R@{k:<3}         {v:.3f}")
        for k, v in sorted(self.ndcg_at_k.items()):
            lines.append(f"  NDCG@{k:<2}       {v:.3f}")
        lines.append(f"  Confidence:   {self.mean_confidence:.3f}")
        lines.append(f"  Avg tokens:   {self.mean_tokens_delivered:.0f}")
        return "\n".join(lines)

    def compare(self, other: "EvalResult") -> dict:
        """Return per-metric deltas (other - self). Positive = other is better."""
        out: dict = {"mrr": other.mrr - self.mrr}
        for k in self.precision_at_k:
            out[f"p@{k}"] = other.precision_at_k.get(k, 0) - self.precision_at_k[k]
        for k in self.recall_at_k:
            out[f"r@{k}"] = other.recall_at_k.get(k, 0) - self.recall_at_k[k]
        for k in self.ndcg_at_k:
            out[f"ndcg@{k}"] = other.ndcg_at_k.get(k, 0) - self.ndcg_at_k[k]
        return out


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------

class RetrievalEvaluator:
    """Evaluates ContextGraph retrieval quality against ground truth."""

    def __init__(self, graph: "ContextGraph") -> None:
        self._graph = graph

    def evaluate(
        self,
        cases: list[EvalCase],
        top_k: int = 10,
        ks: list[int] | None = None,
        layer: str = "l1",
        budget: int = 2000,
        confidence_threshold: float = 0.0,
    ) -> EvalResult:
        """Run evaluation over all cases.

        Args:
            cases:    List of EvalCase with queries and relevant node IDs.
            top_k:    Max nodes to retrieve per query via scan_nodes().
            ks:       k values for P@k, R@k, NDCG@k. Default [1, 3, 5, 10].
            layer:    Retrieval layer for token count measurement.
            budget:   Token budget per node retrieval.
            confidence_threshold: Passed to retrieve() for l1 fallback.
        """
        if ks is None:
            ks = [1, 3, 5, 10]
        ks = [k for k in ks if k <= top_k]

        per_query: list[PerQueryResult] = []
        layer_counter: dict[str, int] = {}

        for case in cases:
            retrieved = self._graph.scan_nodes(case.query, top_k=top_k)
            retrieved_ids = [nid for nid, _ in retrieved]
            retrieved_scores = {nid: score for nid, score in retrieved}
            relevant_set = set(case.relevant_node_ids)

            hits = [nid for nid in retrieved_ids if nid in relevant_set]
            precision = len(hits) / max(len(retrieved_ids), 1)
            recall = len(hits) / max(len(relevant_set), 1)

            # MRR: position of first hit (1-indexed)
            rr = 0.0
            for rank, nid in enumerate(retrieved_ids, 1):
                if nid in relevant_set:
                    rr = 1.0 / rank
                    break

            # NDCG
            ndcg = _ndcg(retrieved_ids, case.relevant_node_ids, top_k)

            # Confidence and token measurement across retrieved nodes
            confidences = []
            tokens_delivered = 0
            for nid in retrieved_ids[:top_k]:
                content, layer_used, conf = self._graph.retrieve(
                    nid, layer, case.query, budget,
                    confidence_threshold=confidence_threshold,
                )
                confidences.append(conf)
                if isinstance(content, list):
                    from ..core.tokens import count_tokens_list
                    tokens_delivered += count_tokens_list(content)
                else:
                    from ..core.tokens import count_tokens
                    tokens_delivered += count_tokens(content)
                layer_counter[layer_used] = layer_counter.get(layer_used, 0) + 1

            per_query.append(PerQueryResult(
                query=case.query,
                retrieved_node_ids=retrieved_ids,
                relevant_node_ids=case.relevant_node_ids,
                hits=hits,
                reciprocal_rank=rr,
                precision=precision,
                recall=recall,
                ndcg=ndcg,
                mean_confidence=sum(confidences) / max(len(confidences), 1),
                tokens_delivered=tokens_delivered,
            ))

        n = max(len(per_query), 1)

        # Aggregate P@k, R@k, NDCG@k
        def mean_at_k(k: int, attr: str) -> float:
            vals = []
            for pq in per_query:
                retrieved_k = pq.retrieved_node_ids[:k]
                rel = set(pq.relevant_node_ids)
                hits_k = [x for x in retrieved_k if x in rel]
                if attr == "precision":
                    vals.append(len(hits_k) / max(k, 1))
                elif attr == "recall":
                    vals.append(len(hits_k) / max(len(rel), 1))
                elif attr == "ndcg":
                    vals.append(_ndcg(pq.retrieved_node_ids, pq.relevant_node_ids, k))
            return sum(vals) / max(len(vals), 1)

        total_layer_calls = max(sum(layer_counter.values()), 1)
        return EvalResult(
            num_cases=len(cases),
            precision_at_k={k: mean_at_k(k, "precision") for k in ks},
            recall_at_k={k: mean_at_k(k, "recall") for k in ks},
            mrr=sum(pq.reciprocal_rank for pq in per_query) / n,
            ndcg_at_k={k: mean_at_k(k, "ndcg") for k in ks},
            mean_confidence=sum(pq.mean_confidence for pq in per_query) / n,
            mean_tokens_delivered=sum(pq.tokens_delivered for pq in per_query) / n,
            layer_distribution={
                lyr: count / total_layer_calls
                for lyr, count in layer_counter.items()
            },
            per_query=per_query,
        )


# ------------------------------------------------------------------
# Bootstrap eval dataset from production usage
# ------------------------------------------------------------------

def dataset_from_store(store: "MemoryStore", min_hits: int = 2) -> list[EvalCase]:
    """Build EvalCase list from MemoryStore session history (no manual labeling).

    Sessions where certain nodes were repeatedly boosted (used()) are treated
    as positive relevance signals. Useful for bootstrapping an eval dataset
    from production traffic.
    """
    sessions = store.all_sessions()
    query_to_nodes: dict[str, set[str]] = {}

    for s in sessions:
        q = s.get("query", "")
        node_ids = s.get("node_ids", [])
        if isinstance(node_ids, str):
            import json
            try:
                node_ids = json.loads(node_ids)
            except Exception:
                node_ids = []
        if q and node_ids:
            query_to_nodes.setdefault(q, set()).update(node_ids)

    cases = []
    for query, node_ids in query_to_nodes.items():
        if len(node_ids) >= min_hits:
            cases.append(EvalCase(
                query=query,
                relevant_node_ids=list(node_ids),
            ))
    return cases


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _dcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel_set = set(relevant)
    dcg = 0.0
    for rank, nid in enumerate(retrieved[:k], 1):
        if nid in rel_set:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def _idcg(relevant: list[str], k: int) -> float:
    idcg = 0.0
    for rank in range(1, min(len(relevant), k) + 1):
        idcg += 1.0 / math.log2(rank + 1)
    return idcg


def _ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    idcg = _idcg(relevant, k)
    if idcg == 0:
        return 0.0
    return _dcg(retrieved, relevant, k) / idcg
