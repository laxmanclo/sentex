"""Tests for telemetry and eval modules."""
import numpy as np
import pytest

from sentex.telemetry.collector import MetricsCollector, OperationMetrics
from sentex.retrieval.eval import RetrievalEvaluator, EvalCase, EvalResult, _ndcg
from sentex.ingestion.embedder import Embedder
from sentex.core.graph import ContextGraph


class _FakeEmbedder(Embedder):
    def __init__(self, dim=16):
        self._dim = dim
        self._model = object()
        self._cache = {}
        self.model_name = "fake"

    def embed(self, texts):
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(self._dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.stack(vecs)

    def embed_one(self, text):
        return self.embed([text])[0]

    @property
    def dim(self):
        return self._dim


# ------------------------------------------------------------------
# MetricsCollector
# ------------------------------------------------------------------

def test_collector_starts_empty():
    c = MetricsCollector()
    assert c.summary() == {}


def test_collector_records_op():
    c = MetricsCollector()
    c.record(OperationMetrics(
        operation="retrieve",
        node_id="resources/x",
        duration_ms=12.5,
        tokens_out=300,
        confidence=0.87,
    ))
    s = c.summary()
    assert "retrieve" in s
    assert s["retrieve"]["count"] == 1
    assert s["retrieve"]["mean_ms"] == 12.5
    assert s["retrieve"]["mean_confidence"] == 0.87


def test_collector_aggregates_multiple():
    c = MetricsCollector()
    for i in range(10):
        c.record(OperationMetrics(
            operation="ingest",
            node_id=f"node/{i}",
            duration_ms=float(i * 10),
        ))
    s = c.summary()
    assert s["ingest"]["count"] == 10
    assert s["ingest"]["mean_ms"] == 45.0   # mean of 0,10,...,90


def test_collector_reset():
    c = MetricsCollector()
    c.record(OperationMetrics(operation="ingest", node_id=None, duration_ms=1.0))
    c.reset()
    assert c.summary() == {}


def test_collector_maxlen():
    c = MetricsCollector(maxlen=5)
    for i in range(10):
        c.record(OperationMetrics(operation="op", node_id=None, duration_ms=float(i)))
    assert len(c.recent(100)) == 5   # buffer capped at maxlen


def test_collector_measure_context_manager():
    c = MetricsCollector()
    with c.measure("retrieve", node_id="x") as m:
        m.tokens_out = 100
        m.confidence = 0.75
    s = c.summary()
    assert "retrieve" in s
    assert s["retrieve"]["mean_tokens_out"] == 100.0
    assert s["retrieve"]["mean_confidence"] == 0.75


def test_graph_with_metrics():
    """ContextGraph accepts a MetricsCollector and records ops."""
    c = MetricsCollector()
    g = ContextGraph(embedder=_FakeEmbedder(), metrics=c)
    g.put("resources/a", "Alpha sentence. Beta sentence.", agent_id="x")
    g.get("resources/a", query="what is alpha", budget=500)
    # retrieve op should be recorded
    s = c.summary()
    assert "retrieve" in s


# ------------------------------------------------------------------
# NDCG helper
# ------------------------------------------------------------------

def test_ndcg_perfect():
    retrieved = ["A", "B", "C"]
    relevant = ["A", "B", "C"]
    assert _ndcg(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_zero():
    retrieved = ["X", "Y"]
    relevant = ["A", "B"]
    assert _ndcg(retrieved, relevant, k=2) == 0.0


def test_ndcg_partial():
    retrieved = ["A", "X", "B"]
    relevant = ["A", "B"]
    score = _ndcg(retrieved, relevant, k=3)
    assert 0.0 < score < 1.0


# ------------------------------------------------------------------
# RetrievalEvaluator
# ------------------------------------------------------------------

@pytest.fixture
def populated_graph():
    g = ContextGraph(embedder=_FakeEmbedder())
    g.put("resources/immune", "T-cells destroy infected cells. B-cells produce antibodies.", agent_id="r")
    g.put("resources/general", "The sky is blue. Water is wet.", agent_id="r")
    g.put("working/analysis", "Key immune mechanisms identified.", agent_id="a")
    return g


def test_evaluator_runs(populated_graph):
    evaluator = RetrievalEvaluator(populated_graph)
    cases = [
        EvalCase(
            query="immune cells and mechanisms",
            relevant_node_ids=["resources/immune"],
        ),
    ]
    result = evaluator.evaluate(cases, top_k=5)
    assert isinstance(result, EvalResult)
    assert result.num_cases == 1
    assert 0.0 <= result.mrr <= 1.0


def test_evaluator_precision_recall_range(populated_graph):
    evaluator = RetrievalEvaluator(populated_graph)
    cases = [
        EvalCase(
            query="antibodies immune system",
            relevant_node_ids=["resources/immune", "working/analysis"],
        ),
    ]
    result = evaluator.evaluate(cases, ks=[1, 3])
    for k, v in result.precision_at_k.items():
        assert 0.0 <= v <= 1.0
    for k, v in result.recall_at_k.items():
        assert 0.0 <= v <= 1.0


def test_evaluator_mrr_range(populated_graph):
    evaluator = RetrievalEvaluator(populated_graph)
    cases = [
        EvalCase(query="T-cells", relevant_node_ids=["resources/immune"]),
        EvalCase(query="weather", relevant_node_ids=["resources/general"]),
    ]
    result = evaluator.evaluate(cases, top_k=5)
    assert 0.0 <= result.mrr <= 1.0


def test_evaluator_summary_string(populated_graph):
    evaluator = RetrievalEvaluator(populated_graph)
    cases = [EvalCase(query="immune", relevant_node_ids=["resources/immune"])]
    result = evaluator.evaluate(cases)
    summary = result.summary()
    assert "MRR" in summary
    assert "P@" in summary


def test_evaluator_compare(populated_graph):
    evaluator = RetrievalEvaluator(populated_graph)
    cases = [EvalCase(query="immune cells", relevant_node_ids=["resources/immune"])]
    r1 = evaluator.evaluate(cases, top_k=3)
    r2 = evaluator.evaluate(cases, top_k=5)
    delta = r1.compare(r2)
    assert "mrr" in delta
    assert isinstance(delta["mrr"], float)


def test_evaluator_per_query(populated_graph):
    evaluator = RetrievalEvaluator(populated_graph)
    cases = [
        EvalCase(query="immune system", relevant_node_ids=["resources/immune"]),
        EvalCase(query="blue sky", relevant_node_ids=["resources/general"]),
    ]
    result = evaluator.evaluate(cases)
    assert len(result.per_query) == 2
    for pq in result.per_query:
        assert 0.0 <= pq.reciprocal_rank <= 1.0
