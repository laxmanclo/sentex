"""Backward-compatibility shim — imports from sentex.core.scoring."""
from .core.scoring import *  # noqa: F401, F403
from .core.scoring import HotnessScore, compute_hotness
