"""Hotness scoring for KNN graph edges.

Replaces the old flat additive boost with a principled frequency-recency model:

    hotness = sigmoid(hit_count / freq_scale) × exp(-ln(2) × age_s / half_life_s)

- freq_term saturates via sigmoid — 10 hits ≈ 0.73, 50 hits ≈ 0.98 (diminishing returns)
- decay_term halves every half_life_s seconds (default 24 h, configurable)

A cold edge (0 hits) starts at hotness = 0.0. After the first hit it jumps to
sigmoid(1/10) ≈ 0.52 × 1.0 (fresh) — immediately influencing retrieval.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class HotnessScore:
    """Per-edge usage record. Mutable — call hit() each time the edge is traversed."""
    hit_count: int = 0
    last_hit_at: float = field(default_factory=time.time)

    def hit(self, now: float | None = None) -> None:
        self.hit_count += 1
        self.last_hit_at = now if now is not None else time.time()

    def to_dict(self) -> dict:
        return {"hit_count": self.hit_count, "last_hit_at": self.last_hit_at}

    @classmethod
    def from_dict(cls, d: dict) -> "HotnessScore":
        return cls(hit_count=d.get("hit_count", 0), last_hit_at=d.get("last_hit_at", time.time()))


def compute_hotness(
    score: HotnessScore,
    now: float | None = None,
    freq_scale: float = 10.0,
    half_life_s: float = 86_400.0,
) -> float:
    """Return a [0, 1) hotness value combining hit frequency and recency decay.

    Args:
        score:        The HotnessScore for the edge.
        now:          Current unix timestamp. Defaults to time.time().
        freq_scale:   Sigmoid midpoint — hit_count == freq_scale → hotness ≈ 0.73.
        half_life_s:  Recency half-life in seconds. Default = 24 hours.

    Returns:
        float in [0, 1): 0 = cold/stale, approaching 1 = hot/recent.
    """
    if score.hit_count == 0:
        return 0.0
    if now is None:
        now = time.time()
    freq_term = 1.0 / (1.0 + math.exp(-score.hit_count / freq_scale))
    age_s = max(0.0, now - score.last_hit_at)
    decay_term = math.exp(-math.log(2) * age_s / half_life_s)
    return freq_term * decay_term
