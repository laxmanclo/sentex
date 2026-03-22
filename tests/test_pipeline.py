"""End-to-end tests for the Pipeline SDK layer."""
from __future__ import annotations

import asyncio
import pytest
import numpy as np

from sentex import Pipeline, Read, Write
from sentex.ingestion.embedder import Embedder
from sentex.core.graph import ContextGraph


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


def _make_pipeline() -> Pipeline:
    graph = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
    return Pipeline(graph=graph)


async def _mock_llm(prompt: str) -> str:
    return f"LLM response to: {prompt[:40]}"


# ------------------------------------------------------------------


def test_single_agent_pipeline():
    pipeline = _make_pipeline()

    @pipeline.agent(id="producer", writes=["data"])
    async def producer(ctx):
        return "The mitochondria is the powerhouse of the cell."

    result = asyncio.run(pipeline.run(query="biology facts", llm=_mock_llm))

    assert "data" in result.outputs
    assert "mitochondria" in result.outputs["data"]
    assert len(result.agent_results) == 1
    assert result.graph.node_count == 1


def test_two_agent_pipeline_data_flows():
    pipeline = _make_pipeline()

    @pipeline.agent(id="agent1", writes=["facts"])
    async def agent1(ctx):
        return "Photosynthesis converts sunlight into glucose. Chlorophyll absorbs light energy."

    @pipeline.agent(
        id="agent2",
        reads=[Read("facts", layer="l1", budget=500)],
        writes=["summary"],
    )
    async def agent2(ctx):
        assert "facts" in ctx
        content = ctx["facts"]
        assert isinstance(content, (str, list))
        return "Summary: plants use sunlight."

    result = asyncio.run(pipeline.run(query="how do plants work?", llm=_mock_llm))

    assert "facts" in result.outputs
    assert "summary" in result.outputs
    assert result.graph.node_count == 2


def test_seed_context_available_to_first_agent():
    pipeline = _make_pipeline()

    @pipeline.agent(
        id="responder",
        reads=[Read("user-query", layer="l3")],
        writes=["response"],
    )
    async def responder(ctx):
        query_text = ctx["user-query"]
        return f"Answer to: {query_text}"

    result = asyncio.run(pipeline.run(
        query="test",
        llm=_mock_llm,
        seed_context={"user-query": "What is quantum entanglement?"},
    ))

    assert "response" in result.outputs
    assert "quantum" in result.outputs["response"]


def test_pipeline_validation_catches_missing_read():
    pipeline = _make_pipeline()

    @pipeline.agent(
        id="reader",
        reads=[Read("does-not-exist", layer="l1", budget=500)],
        writes=["output"],
    )
    async def reader(ctx):
        return "done"

    with pytest.raises(ValueError, match="validation failed"):
        asyncio.run(pipeline.run(query="test", llm=_mock_llm))


def test_validation_passes_with_correct_order():
    pipeline = _make_pipeline()

    @pipeline.agent(id="a1", writes=["node-a"])
    async def a1(ctx):
        return "content from a1"

    @pipeline.agent(id="a2", reads=[Read("node-a", layer="l1", budget=500)], writes=["node-b"])
    async def a2(ctx):
        return "content from a2"

    errors = pipeline.validate()
    assert errors == []


def test_agent_context_render():
    pipeline = _make_pipeline()
    rendered_output = {}

    @pipeline.agent(id="p", writes=["text"])
    async def p(ctx):
        return "The sky is blue. The ocean is deep. Stars are far away."

    @pipeline.agent(id="c", reads=[Read("text", layer="l1", budget=500)], writes=["out"])
    async def c(ctx):
        rendered = ctx.render()
        rendered_output["r"] = rendered
        return "done"

    asyncio.run(pipeline.run(query="nature facts", llm=_mock_llm))

    assert "text" in rendered_output.get("r", "")
    assert "===" in rendered_output.get("r", "")


def test_pipeline_result_summary():
    pipeline = _make_pipeline()

    @pipeline.agent(id="solo", writes=["out"])
    async def solo(ctx):
        return "hello"

    result = asyncio.run(pipeline.run(query="hi", llm=_mock_llm))
    summary = result.summary()
    assert "solo" in summary
    assert "tok" in summary


def test_sync_llm_callable():
    """Pipeline accepts a plain sync callable as llm=."""
    pipeline = _make_pipeline()

    def sync_llm(prompt: str) -> str:
        return "sync response"

    @pipeline.agent(id="a", writes=["out"])
    async def a(ctx):
        return await ctx.llm("test prompt")

    result = asyncio.run(pipeline.run(query="q", llm=sync_llm))
    assert result.outputs["out"] == "sync response"


def test_three_agent_chain():
    pipeline = _make_pipeline()
    call_order = []

    @pipeline.agent(id="first", writes=["raw"])
    async def first(ctx):
        call_order.append("first")
        return "Raw data: apples oranges bananas are fruits."

    @pipeline.agent(
        id="second",
        reads=[Read("raw", layer="l1", budget=300)],
        writes=["processed"],
    )
    async def second(ctx):
        call_order.append("second")
        return "Processed: fruits identified."

    @pipeline.agent(
        id="third",
        reads=[
            Read("raw", layer="l1", budget=200),
            Read("processed", layer="l3"),
        ],
        writes=["final"],
    )
    async def third(ctx):
        call_order.append("third")
        return "Final output complete."

    result = asyncio.run(pipeline.run(query="classify fruits", llm=_mock_llm))

    assert call_order == ["first", "second", "third"]
    assert "final" in result.outputs
    assert result.graph.node_count == 3
