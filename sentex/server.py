"""FastAPI HTTP server — POST /assemble.

Allows TypeScript (or any HTTP client) to use Engram as a service.

Start with:  uvicorn engram.server:app --port 8765
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .graph import ContextGraph
from .manifest import defineAgent
from .types import Read, Write

app = FastAPI(title="Engram", version="0.1.0")

# Single shared graph instance for the server lifetime
_graph = ContextGraph()


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------


class ReadModel(BaseModel):
    node_id: str
    layer: str = "l1"
    budget: int = 2000


class IngestRequest(BaseModel):
    node_id: str
    content: str
    agent_id: str


class AssembleRequest(BaseModel):
    agent_id: str
    reads: list[ReadModel]
    token_budget: int = 4000
    fallback: str = "l2"
    confidence_threshold: float = 0.5
    query: str


class MarkUsedRequest(BaseModel):
    context_token: str  # unused — for future auth
    assembled_node_ids: list[str]
    used_ids: list[str]


class ValidateRequest(BaseModel):
    agents: list[dict[str, Any]]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    node = await _graph.ingest_async(req.node_id, req.content, req.agent_id)
    return {"node_id": node.id, "sentences": len(node.sentence_ids)}


@app.post("/assemble")
def assemble(req: AssembleRequest) -> dict:
    reads = [Read(r.node_id, r.layer, r.budget) for r in req.reads]
    agent = defineAgent(
        id=req.agent_id,
        reads=reads,
        writes=[],
        token_budget=req.token_budget,
        fallback=req.fallback,
        confidence_threshold=req.confidence_threshold,
    )
    assembled = _graph.assemble_for(agent, req.query)
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


@app.post("/mark_used")
def mark_used(req: MarkUsedRequest) -> dict:
    # Reconstruct a minimal AssembledContext for the call
    from .types import AssembledContext

    dummy = AssembledContext(
        context={},
        token_count=0,
        budget=0,
        utilization=0.0,
        layers_used={},
        compressed=[],
        missing=[],
        confidence={},
    )
    _graph.mark_used(dummy, req.used_ids)
    return {"ok": True}


@app.get("/nodes")
def list_nodes() -> dict:
    return {
        "nodes": [
            {
                "id": n.id,
                "produced_by": n.produced_by,
                "sentences": len(n.sentence_ids),
                "l0": n.l0,
            }
            for n in _graph._nodes.values()
        ]
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "nodes": _graph.node_count,
        "sentences": _graph.sentence_count,
    }
