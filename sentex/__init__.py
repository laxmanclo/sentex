"""Sentex — sentence-graph context management for multi-agent AI pipelines."""

from .graph import ContextGraph
from .manifest import defineAgent
from .persistence import load, save
from .pipeline import Pipeline, PipelineResult, AgentResult
from .context import AgentContext
from .store import MemoryStore
from .session import SessionRecord
from .types import (
    AgentManifest,
    AssembledContext,
    ContextNode,
    Read,
    Write,
)

__all__ = [
    # Primary SDK — this is what you import
    "Pipeline",
    "Read",
    "Write",
    # Cross-run memory
    "MemoryStore",
    # Lower-level (advanced use)
    "ContextGraph",
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

__version__ = "0.1.0"
