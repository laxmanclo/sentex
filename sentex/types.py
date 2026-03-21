"""Core data structures for Sentex."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SentenceMetadata:
    sentence_idx: int   # index into ContextGraph.sentences / .embeddings
    node_id: str        # which ContextNode this sentence belongs to
    agent_id: str       # which agent produced it


@dataclass
class L0L1L2L3TokenCounts:
    l0: int = 0
    l1: int = 0   # sum of all sentences
    l2: int = 0
    l3: int = 0


@dataclass
class ContextNode:
    id: str
    produced_by: str            # agent id that wrote this node
    l3: Any                     # full raw content
    sentence_ids: list[int]     # indices into graph-level arrays
    l0: str = ""                # ~50 token identity sentence (LLM-generated or extractive)
    l2: str = ""                # ~2k summary (LLM-generated or extractive)
    first_sentence: str = ""    # first sentence of content — used as extractive L0 fallback
    token_counts: L0L1L2L3TokenCounts = field(default_factory=L0L1L2L3TokenCounts)


@dataclass
class Read:
    node_id: str
    layer: str = "l1"           # "l0" | "l1" | "l2" | "l3"
    budget: int = 2000          # token budget for l1 retrieval


@dataclass
class AutoRead:
    """Dynamic read — scan all nodes at L0, retrieve top-k at the declared layer.

    Use when you don't know node IDs at pipeline-definition time.

    Example:
        AutoRead(top_k=3, layer="l1", budget_per_node=1000, scope="resources")
        → scan all resources/* nodes at L0
        → retrieve top-3 most relevant at L1
        → returned in context as {"auto:resources/search": [...], ...}
    """
    top_k: int = 3
    layer: str = "l1"
    budget_per_node: int = 1000
    scope: str | None = None    # e.g. "resources" to only scan resources/* nodes


@dataclass
class Write:
    node_id: str


@dataclass
class AgentManifest:
    id: str
    reads: list[Read | AutoRead]
    writes: list[Write]
    token_budget: int
    fallback: str = "l2"        # layer to fall back to if l1 confidence low
    confidence_threshold: float = 0.5


@dataclass
class AssembledContext:
    context: dict[str, str | list[str]]
    token_count: int
    budget: int
    utilization: float
    layers_used: dict[str, str]
    compressed: list[str]       # nodes that fell back to a lower layer
    missing: list[str]          # declared reads not yet in graph
    confidence: dict[str, float]
