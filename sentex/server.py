"""Sentex FastAPI server.

Exposes the ContextGraph over HTTP so TypeScript, Go, or any other
language can use Sentex as a context management service.

Start:
    uvicorn sentex.server:app --port 8765

Environment:
    SENTEX_PERSIST   Path to SQLite file for cross-run memory (optional).
                     If set, edge weights and summaries persist across restarts.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .graph import ContextGraph
from .manifest import defineAgent
from .types import AutoRead, Read


# ------------------------------------------------------------------
# App state — graph is created once, optionally backed by MemoryStore
# ------------------------------------------------------------------

class _State:
    graph: ContextGraph


state = _State()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    persist = os.getenv("SENTEX_PERSIST")
    if persist:
        from .store import MemoryStore
        store = MemoryStore(persist)
        state.graph = ContextGraph()
        state.graph._store = store
    else:
        state.graph = ContextGraph()
    yield


app = FastAPI(title="Sentex", version="0.1.0", lifespan=_lifespan)


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class PutRequest(BaseModel):
    node_id: str
    content: str
    agent_id: str = "agent"


class GetRequest(BaseModel):
    node_id: str
    query: str
    budget: int = 2000
    layer: str = "l1"


class ScanRequest(BaseModel):
    query: str
    top_k: int = 5
    scope: str | None = None


class ReadModel(BaseModel):
    node_id: str
    layer: str = "l1"
    budget: int = 2000


class AssembleRequest(BaseModel):
    agent_id: str
    reads: list[ReadModel]
    token_budget: int = 4000
    fallback: str = "l2"
    confidence_threshold: float = 0.5
    query: str


class MarkUsedRequest(BaseModel):
    node_ids: list[str]


# ------------------------------------------------------------------
# Simple API — put / get / used / scan
# ------------------------------------------------------------------

@app.post("/put", summary="Store agent output in the graph")
async def put(req: PutRequest) -> dict:
    """Store an agent's output. Sentences are split, embedded, and added to the KNN graph."""
    node = state.graph.ingest(req.node_id, req.content, agent_id=req.agent_id)
    return {
        "node_id": node.id,
        "sentences": len(node.sentence_ids),
        "first_sentence": node.first_sentence,
    }


@app.post("/get", summary="Retrieve context for a node")
def get(req: GetRequest) -> dict:
    """Retrieve context from a node at the specified layer and budget."""
    if req.node_id not in state.graph._nodes:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found")

    content = state.graph.get(req.node_id, query=req.query, budget=req.budget, layer=req.layer)
    _, layer_used, confidence = state.graph.retrieve(
        req.node_id, req.layer, req.query, req.budget
    )
    return {
        "node_id": req.node_id,
        "content": content,
        "layer_used": layer_used,
        "confidence": confidence,
        "is_list": isinstance(content, list),
    }


@app.post("/used", summary="Mark a node as used to boost future retrieval")
def used(req: MarkUsedRequest) -> dict:
    """Boost edge weights for nodes that were useful. Improves future retrieval."""
    for node_id in req.node_ids:
        state.graph.used(node_id)
    return {"ok": True, "boosted": req.node_ids}


@app.post("/scan", summary="Find the most relevant nodes for a query")
def scan(req: ScanRequest) -> dict:
    """Scan all nodes at L0 and return the top-k most relevant to the query."""
    results = state.graph.scan_nodes(req.query, top_k=req.top_k, scope=req.scope)
    return {
        "results": [{"node_id": nid, "score": score} for nid, score in results]
    }


# ------------------------------------------------------------------
# Structured API — assemble / mark_used
# ------------------------------------------------------------------

@app.post("/assemble", summary="Assemble context for an agent manifest")
def assemble(req: AssembleRequest) -> dict:
    """Assemble context across multiple reads with budget enforcement and fallback."""
    reads = [Read(r.node_id, r.layer, r.budget) for r in req.reads]
    agent = defineAgent(
        id=req.agent_id,
        reads=reads,
        writes=[],
        token_budget=req.token_budget,
        fallback=req.fallback,
        confidence_threshold=req.confidence_threshold,
    )
    assembled = state.graph.assemble_for(agent, req.query)
    return {
        "context": assembled.context,
        "token_count": assembled.token_count,
        "budget": assembled.budget,
        "utilization": assembled.utilization,
        "layers_used": assembled.layers_used,
        "compressed": assembled.compressed,
        "missing": assembled.missing,
        "confidence": assembled.confidence,
    }


@app.post("/mark_used", summary="Mark assembled nodes as used")
def mark_used(req: MarkUsedRequest) -> dict:
    from .types import AssembledContext
    dummy = AssembledContext(
        context={nid: [] for nid in req.node_ids},
        token_count=0, budget=0, utilization=0.0,
        layers_used={}, compressed=[], missing=[], confidence={},
    )
    state.graph.mark_used(dummy, req.node_ids)
    return {"ok": True}


# ------------------------------------------------------------------
# Inspection
# ------------------------------------------------------------------

@app.get("/nodes", summary="List all nodes in the graph")
def list_nodes() -> dict:
    return {
        "nodes": [
            {
                "id": n.id,
                "produced_by": n.produced_by,
                "sentences": len(n.sentence_ids),
                "l0": n.l0 or n.first_sentence,
                "has_l2": bool(n.l2),
                "token_counts": {
                    "l0": n.token_counts.l0,
                    "l1": n.token_counts.l1,
                    "l2": n.token_counts.l2,
                    "l3": n.token_counts.l3,
                },
            }
            for n in state.graph._nodes.values()
        ]
    }


@app.get("/nodes/{node_id:path}", summary="Inspect a single node")
def get_node(node_id: str) -> dict:
    node = state.graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return {
        "id": node.id,
        "produced_by": node.produced_by,
        "sentences": len(node.sentence_ids),
        "l0": node.l0 or node.first_sentence,
        "l2": node.l2,
        "has_l3": node.l3 is not None,
        "first_sentence": node.first_sentence,
        "token_counts": {
            "l0": node.token_counts.l0,
            "l1": node.token_counts.l1,
            "l2": node.token_counts.l2,
            "l3": node.token_counts.l3,
        },
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "nodes": state.graph.node_count,
        "sentences": state.graph.sentence_count,
        "persist": bool(os.getenv("SENTEX_PERSIST")),
    }
