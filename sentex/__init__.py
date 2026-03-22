"""Sentex — sentence-graph context management for multi-agent AI pipelines."""

from .graph import ContextGraph
from .manifest import defineAgent
from .persistence import load, save
from .pipeline import Pipeline, PipelineResult, AgentResult
from .context import AgentContext
from .store import MemoryStore
from .session import SessionRecord
from .scoring import HotnessScore, compute_hotness
from .telemetry import MetricsCollector, OperationMetrics, make_prometheus_exporter
from .relations import Relation, RelationIndex
from .fs import GraphFS, FSEntry
from .eval import RetrievalEvaluator, EvalCase, EvalResult, dataset_from_store
from .types import (
    AgentManifest,
    AssembledContext,
    AutoRead,
    ContextNode,
    Read,
    Write,
)

__all__ = [
    # Primary SDK — BYOA interface
    "ContextGraph",
    "Pipeline",
    "Read",
    "AutoRead",
    "Write",
    # Cross-run memory
    "MemoryStore",
    # Telemetry
    "MetricsCollector",
    "OperationMetrics",
    "make_prometheus_exporter",
    # Eval
    "RetrievalEvaluator",
    "EvalCase",
    "EvalResult",
    "dataset_from_store",
    # Relations
    "Relation",
    "RelationIndex",
    # Filesystem API
    "GraphFS",
    "FSEntry",
    # Hotness scoring
    "HotnessScore",
    "compute_hotness",
    # Lower-level (advanced use)
    "defineAgent",
    "AgentContext",
    # Persistence
    "load",
    "save",
    # Types
    "AgentManifest",
    "AssembledContext",
    "AgentResult",
    "PipelineResult",
    "SessionRecord",
    "ContextNode",
]

__version__ = "0.2.1"
