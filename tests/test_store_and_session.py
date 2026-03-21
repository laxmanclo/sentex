"""Tests for MemoryStore, sessions, scoped nodes, and cross-run memory."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sentex import Pipeline, Read, MemoryStore
from sentex.embedder import Embedder
from sentex.graph import ContextGraph
from sentex.pipeline import _extractive_summary, _scope


class _FakeEmbedder(Embedder):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._model = object()
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


async def _mock_llm(prompt: str) -> str:
    return "mock response"


# ------------------------------------------------------------------
# Scope parsing
# ------------------------------------------------------------------

def test_scope_parsing():
    assert _scope("resources/search") == "resources"
    assert _scope("working/script") == "working"
    assert _scope("memory/prefs") == "memory"
    assert _scope("flat-node") == "working"


# ------------------------------------------------------------------
# Extractive L2 fallback
# ------------------------------------------------------------------

def test_extractive_summary_returns_content():
    text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."
    summary = _extractive_summary(text, max_tokens=20)
    assert len(summary) > 0
    assert "Alpha" in summary


def test_extractive_summary_respects_budget():
    # Long text — summary should be shorter than input
    text = " ".join(["This is sentence number {}.".format(i) for i in range(200)])
    summary = _extractive_summary(text, max_tokens=50)
    from sentex.tokens import count_tokens
    assert count_tokens(summary) <= 55  # small tolerance


def test_l2_never_empty_after_ingest():
    """After pipeline runs an agent, the written node must never have l2 = ''."""
    graph = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
    pipeline = Pipeline(graph=graph)

    @pipeline.agent(id="producer", writes=["working/data"])
    async def producer(ctx):
        return "T-cells fight infection. B-cells produce antibodies."

    asyncio.run(pipeline.run(query="immune system", llm=_mock_llm))

    node = graph.get_node("working/data")
    assert node is not None
    assert node.l2 != ""


# ------------------------------------------------------------------
# MemoryStore
# ------------------------------------------------------------------

def test_memory_store_save_and_load_summary():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "test.db")
        store.save_node_summary("resources", "search", "L0 text", "L2 text", "raw", "agent-1")

        result = store.load_node_summary("resources", "search")
        assert result is not None
        assert result["l0"] == "L0 text"
        assert result["l2"] == "L2 text"


def test_memory_store_write_and_read_memory():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "test.db")
        store.write_memory("memory", "user_pref", "prefers bullet points", "run-001")

        result = store.read_memory("memory")
        assert "user_pref" in result
        assert result["user_pref"] == "prefers bullet points"


def test_memory_store_session_tracking():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "test.db")
        store.record_session("s1", "test query", ["a1", "a2"], ["node1"])
        store.commit_session("s1")

        sessions = store.all_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["committed_at"] is not None


def test_memory_store_edge_boost_persist():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "test.db")
        store.boost_edge(1, 2, 0.1)
        store.boost_edge(1, 2, 0.1)
        store.flush_edge_boosts()

        boosts = store.load_edge_boosts()
        assert (1, 2) in boosts
        assert abs(boosts[(1, 2)] - 0.2) < 1e-5


# ------------------------------------------------------------------
# Cross-run memory (persist=)
# ------------------------------------------------------------------

def test_pipeline_persist_records_session():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "sentex.db")
        graph = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
        pipeline = Pipeline(graph=graph, persist=db)

        @pipeline.agent(id="agent1", writes=["working/out"])
        async def agent1(ctx):
            return "Some output from agent one."

        asyncio.run(pipeline.run(query="test", llm=_mock_llm, session_id="test-run-1"))

        history = pipeline.history()
        assert len(history) == 1
        assert history[0]["session_id"] == "test-run-1"
        assert history[0]["committed_at"] is not None


def test_pipeline_persist_caches_l2():
    """Second run should load cached l2 from first run."""
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "sentex.db")

        # First run
        graph1 = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
        p1 = Pipeline(graph=graph1, persist=db)

        @p1.agent(id="a", writes=["resources/data"])
        async def a1(ctx):
            return "Photosynthesis uses sunlight. Chlorophyll absorbs light."

        asyncio.run(p1.run(query="plants", llm=_mock_llm))
        node1 = graph1.get_node("resources/data")
        assert node1.l2 != ""

        # Second run — l2 should be loaded from cache
        graph2 = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
        p2 = Pipeline(graph=graph2, persist=db)

        @p2.agent(id="a", writes=["resources/data"])
        async def a2(ctx):
            return "Photosynthesis uses sunlight. Chlorophyll absorbs light."

        asyncio.run(p2.run(query="plants", llm=_mock_llm))
        node2 = graph2.get_node("resources/data")
        assert node2 is not None
        assert node2.l2 != ""


def test_session_id_in_result():
    graph = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
    pipeline = Pipeline(graph=graph)

    @pipeline.agent(id="x", writes=["out"])
    async def x(ctx):
        return "done"

    result = asyncio.run(pipeline.run(
        query="q", llm=_mock_llm, session_id="my-custom-session"
    ))
    assert result.session_id == "my-custom-session"


def test_scoped_node_ids_work_normally():
    """Scoped node IDs like 'resources/search' are just strings — work exactly like flat ones."""
    graph = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
    pipeline = Pipeline(graph=graph)

    @pipeline.agent(id="searcher", writes=["resources/search"])
    async def searcher(ctx):
        return "Quantum entanglement links particles instantly across distance."

    @pipeline.agent(
        id="writer",
        reads=[Read("resources/search", layer="l1", budget=500)],
        writes=["working/answer"],
    )
    async def writer(ctx):
        assert "resources/search" in ctx
        return "Summary done."

    result = asyncio.run(pipeline.run(query="quantum", llm=_mock_llm))
    assert "resources/search" in result.outputs
    assert "working/answer" in result.outputs
