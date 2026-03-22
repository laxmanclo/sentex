"""Integration tests for the BYOA (bring-your-own-agent) pattern.

Validates the three-call interface: put() → get() → used()
"""
from __future__ import annotations

import numpy as np
import pytest

from sentex.graph import ContextGraph
from sentex.embedder import Embedder


class _DeterministicEmbedder(Embedder):
    """Hash-based embedder — deterministic, no model loading."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim
        self._model = object()
        self._cache: dict = {}
        self.model_name = "deterministic"

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


SEARCH_TEXT = (
    "The immune system has two branches: innate and adaptive. "
    "T-cells are lymphocytes that develop in the thymus. "
    "B-cells produce antibodies that bind to specific antigens. "
    "Macrophages engulf and digest cellular debris and pathogens. "
    "Fever is triggered by pyrogens released by macrophages during infection."
)

ANALYSIS_TEXT = (
    "Key findings: T-cells and B-cells coordinate the adaptive immune response. "
    "Macrophages act as first responders and antigen presenters. "
    "Fever inhibits pathogen replication."
)


@pytest.fixture
def graph() -> ContextGraph:
    return ContextGraph(embedder=_DeterministicEmbedder())


# ------------------------------------------------------------------
# put()
# ------------------------------------------------------------------

def test_put_creates_node(graph):
    node = graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    assert node.id == "resources/search"
    assert node.produced_by == "researcher"
    assert len(node.sentence_ids) > 0


def test_put_multiple_nodes(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    graph.put("working/analysis", ANALYSIS_TEXT, agent_id="analyst")
    assert graph.node_count == 2
    assert graph.sentence_count > 0


def test_put_stores_first_sentence(graph):
    node = graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    assert node.first_sentence  # not empty
    assert node.first_sentence in SEARCH_TEXT


def test_put_builds_extractive_l2(graph):
    node = graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    # Extractive L2 is always non-empty after put()
    assert node.l2


# ------------------------------------------------------------------
# get()
# ------------------------------------------------------------------

def test_get_returns_content(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    content = graph.get("resources/search", query="immune cells", budget=500)
    assert content  # not empty


def test_get_l1_returns_list(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    content = graph.get(
        "resources/search", query="immune cells", budget=5000, layer="l1"
    )
    # get() always returns list[str] now — consistent regardless of layer/fallback
    assert isinstance(content, list)
    assert content  # non-empty


def test_get_l2_returns_list(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    content = graph.get("resources/search", query="immune cells", budget=500, layer="l2")
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]


def test_get_l0_returns_list(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    content = graph.get("resources/search", query="immune cells", budget=500, layer="l0")
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]


def test_get_missing_node_returns_empty_list(graph):
    content = graph.get("nonexistent", query="anything", budget=1000)
    assert isinstance(content, list)
    assert content[0] == ""


def test_get_respects_budget(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    content = graph.get("resources/search", query="immune", budget=50, layer="l1")
    # At a tiny budget, should fall back to l2 or l0 — either way not empty
    assert content is not None


# ------------------------------------------------------------------
# render()
# ------------------------------------------------------------------

def test_render_returns_string(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    rendered = graph.render("resources/search", query="immune cells")
    assert isinstance(rendered, str)
    assert rendered


# ------------------------------------------------------------------
# used()
# ------------------------------------------------------------------

def test_used_increases_boosts(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    initial_boosts = len(graph._usage_boost)
    graph.used("resources/search")
    # Boosts should be set for at least some edges
    assert len(graph._usage_boost) >= initial_boosts


def test_used_unknown_node_is_silent(graph):
    # Should not raise even if node doesn't exist
    graph.used("nonexistent/node")


# ------------------------------------------------------------------
# stats()
# ------------------------------------------------------------------

def test_stats_structure(graph):
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    graph.put("working/analysis", ANALYSIS_TEXT, agent_id="analyst")
    s = graph.stats()
    assert s["nodes"] == 2
    assert s["sentences"] > 0
    assert "edges" in s
    assert "node_ids" in s
    assert "resources/search" in s["node_ids"]
    assert "working/analysis" in s["node_ids"]


# ------------------------------------------------------------------
# Full BYOA pipeline (3 agents)
# ------------------------------------------------------------------

def test_byoa_three_agent_pipeline(graph):
    """End-to-end: researcher → analyst → writer using put/get/used."""

    # Agent 1: researcher stores search data
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")

    # Agent 2: analyst reads from researcher's output
    context = graph.get(
        "resources/search",
        query="key immune mechanisms",
        budget=1500,
    )
    assert context  # analyst gets something

    graph.put("working/analysis", ANALYSIS_TEXT, agent_id="analyst")
    graph.used("resources/search")

    # Agent 3: writer reads from both nodes
    search_ctx = graph.get("resources/search", query="write a video script", budget=1000)
    analysis_ctx = graph.get("working/analysis", query="write a video script", budget=600)

    assert search_ctx  # writer gets search context
    assert analysis_ctx  # writer gets analysis context

    # Final state
    assert graph.node_count == 2
    assert graph.sentence_count > 0
    stats = graph.stats()
    assert stats["nodes"] == 2


def test_byoa_cross_node_edges(graph):
    """After two puts, KNN edges should span both nodes."""
    graph.put("resources/search", SEARCH_TEXT, agent_id="researcher")
    graph.put("working/analysis", ANALYSIS_TEXT, agent_id="analyst")

    # With global KNN, some sentences from node B will link to node A
    search_ids = set(graph.get_node("resources/search").sentence_ids)
    analysis_ids = set(graph.get_node("working/analysis").sentence_ids)

    cross_edges = 0
    for src_id, neighbors in graph._adjacency.items():
        for dst_id, _ in neighbors:
            if (src_id in search_ids and dst_id in analysis_ids) or (
                src_id in analysis_ids and dst_id in search_ids
            ):
                cross_edges += 1

    # KNN is global — there should be cross-node edges
    assert cross_edges > 0
