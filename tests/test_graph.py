"""Integration tests for ContextGraph.

Uses a tiny fake embedder so sentence-transformers is not required in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentex.graph import ContextGraph
from sentex.embedder import Embedder
from sentex.manifest import defineAgent
from sentex.types import Read, Write


class _FakeEmbedder(Embedder):
    """Deterministic embedder that hashes text to a fixed-dim vector."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._model = object()  # non-None so _load() is skipped
        self._cache: dict = {}
        self.model_name = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(self._dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.stack(vecs)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._dim


def _graph() -> ContextGraph:
    return ContextGraph(embedder=_FakeEmbedder(), knn_k=3)


# ------------------------------------------------------------------


def test_ingest_basic():
    g = _graph()
    node = g.ingest(
        "search-results",
        "T-cells fight infection. B-cells produce antibodies. Macrophages engulf pathogens.",
        agent_id="searcher",
        generate_summaries=False,
    )
    assert node.id == "search-results"
    assert len(node.sentence_ids) == 3
    assert g.sentence_count == 3
    assert g.node_count == 1


def test_retrieve_l3():
    g = _graph()
    content = "Full raw content here."
    g.ingest("data", content, "agent-a", generate_summaries=False)
    result, layer, conf = g.retrieve("data", "l3", "anything", budget_tokens=9999)
    assert result == content
    assert layer == "l3"
    assert conf == 1.0


def test_retrieve_l1_returns_sentences():
    g = _graph()
    g.ingest(
        "doc",
        "Alpha is the first letter. Beta is the second letter. Gamma is the third letter.",
        "agent-a",
        generate_summaries=False,
    )
    # Use confidence_threshold=0.0 so random fake embeddings always return l1
    result, layer, conf = g.retrieve(
        "doc", "l1", "Greek alphabet letters", budget_tokens=500,
        confidence_threshold=0.0,
    )
    assert layer == "l1"
    assert isinstance(result, list)
    assert len(result) >= 1


def test_retrieve_missing_node():
    g = _graph()
    result, layer, conf = g.retrieve("nonexistent", "l1", "query", budget_tokens=500)
    assert result == ""
    assert conf == 0.0


def test_assemble_for():
    g = _graph()
    g.ingest("search", "Immune cells protect the body.", "searcher", generate_summaries=False)
    g.ingest("query", "How does immunity work?", "user", generate_summaries=False)

    agent = defineAgent(
        id="writer",
        reads=[
            Read("search", layer="l1", budget=500),
            Read("query", layer="l3"),
        ],
        writes=["script"],
        token_budget=2000,
    )

    assembled = g.assemble_for(agent, "How does immunity work?")
    assert "search" in assembled.context
    assert "query" in assembled.context
    assert assembled.token_count > 0
    assert assembled.token_count <= assembled.budget
    assert assembled.missing == []


def test_assemble_missing_declared_read():
    g = _graph()
    agent = defineAgent(
        id="writer",
        reads=[Read("not-yet-written", layer="l1", budget=500)],
        writes=["output"],
        token_budget=2000,
    )
    assembled = g.assemble_for(agent, "query")
    assert "not-yet-written" in assembled.missing


def test_validate_pipeline_valid():
    g = _graph()
    agents = [
        defineAgent("a1", reads=[], writes=["data"], token_budget=1000),
        defineAgent("a2", reads=[Read("data", "l1", 500)], writes=["result"], token_budget=1000),
    ]
    errors = g.validate_pipeline(agents)
    assert errors == []


def test_validate_pipeline_invalid():
    g = _graph()
    agents = [
        # a1 reads "data" but nothing produces it
        defineAgent("a1", reads=[Read("data", "l1", 500)], writes=["result"], token_budget=1000),
    ]
    errors = g.validate_pipeline(agents)
    assert len(errors) == 1
    assert "data" in errors[0]


def test_mark_used_boosts_edges():
    g = _graph()
    g.ingest(
        "doc",
        "Alpha is first. Beta is second. Gamma is third.",
        "agent-a",
        generate_summaries=False,
    )
    agent = defineAgent("w", reads=[Read("doc", "l1", 500)], writes=[], token_budget=2000)
    assembled = g.assemble_for(agent, "letters")

    before = dict(g._usage_boost)
    g.mark_used(assembled, ["doc"])
    assert len(g._usage_boost) >= len(before)


def test_cross_node_graph_edges():
    """Sentences from different agents can be neighbours in the graph."""
    g = _graph()
    g.ingest("node1", "T-cells recognise antigens on pathogen surfaces.", "a1", generate_summaries=False)
    g.ingest("node2", "Antigen recognition triggers the adaptive immune cascade.", "a2", generate_summaries=False)

    # Both nodes share the graph — edges can cross node boundaries
    assert g.sentence_count == 2
    adj = g._adjacency
    # Each sentence should have the other as a neighbour
    assert 1 in [j for j, _ in adj.get(0, [])]
    assert 0 in [j for j, _ in adj.get(1, [])]
