"""Session — a named, tracked pipeline run.

A Session wraps a Pipeline run with:
  - A unique session_id
  - Recording to MemoryStore (if configured)
  - commit() that persists edge weights + extracted memories

Usage:
    pipeline = Pipeline(persist="./sentex.db")

    result = await pipeline.run(
        query="...",
        llm="gpt-4o",
        session_id="run-2026-03-22",   # optional, auto-generated if omitted
    )

    # After run completes, commit() is called automatically.
    # Edge weights are saved. Next run retrieves better context.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .pipeline import PipelineResult


@dataclass
class SessionRecord:
    session_id: str
    query: str
    started_at: float
    committed_at: float | None = None
    agent_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    result: PipelineResult | None = None

    def duration_s(self) -> float | None:
        if self.committed_at:
            return self.committed_at - self.started_at
        return None

    def summary(self) -> str:
        lines = [
            f"Session {self.session_id}",
            f"  query:    {self.query}",
            f"  agents:   {', '.join(self.agent_ids)}",
            f"  nodes:    {', '.join(self.node_ids)}",
        ]
        if self.committed_at:
            lines.append(f"  duration: {self.duration_s():.1f}s")
        if self.result:
            lines.append(self.result.summary())
        return "\n".join(lines)
