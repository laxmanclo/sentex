"""Agent manifest definition API.

Usage:
    from sentex import defineAgent, Read, Write

    agent = defineAgent(
        id="script-writer",
        reads=[Read("search-results", layer="l1", budget=2000)],
        writes=["script"],
        token_budget=4000,
        fallback="l2",
    )
"""
from __future__ import annotations

from ..core.types import AgentManifest, AutoRead, Read, Write


def defineAgent(
    id: str,
    reads: list[Read | AutoRead],
    writes: list[str | Write],
    token_budget: int = 4000,
    fallback: str = "l2",
    confidence_threshold: float = 0.5,
) -> AgentManifest:
    """Define an agent manifest (the developer-facing API)."""
    normalised_writes = [
        w if isinstance(w, Write) else Write(node_id=w) for w in writes
    ]
    return AgentManifest(
        id=id,
        reads=reads,
        writes=normalised_writes,
        token_budget=token_budget,
        fallback=fallback,
        confidence_threshold=confidence_threshold,
    )
