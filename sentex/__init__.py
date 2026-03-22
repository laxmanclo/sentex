"""Sentex — sentence-graph context management for multi-agent AI pipelines."""

from .core.graph import ContextGraph
from .core.scoring import HotnessScore, compute_hotness
from .core.relations import Relation, RelationIndex
from .core.fs import GraphFS, FSEntry
from .core.types import (
    AgentManifest,
    AssembledContext,
    AutoRead,
    ContextNode,
    Read,
    Write,
)
from .pipeline.pipeline import Pipeline, PipelineResult, AgentResult
from .pipeline.manifest import defineAgent
from .pipeline.context import AgentContext
from .pipeline.session import SessionRecord
from .storage.store import MemoryStore
from .storage.persistence import load, save
from .telemetry.collector import MetricsCollector, OperationMetrics, make_prometheus_exporter
from .retrieval.eval import RetrievalEvaluator, EvalCase, EvalResult, dataset_from_store

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
