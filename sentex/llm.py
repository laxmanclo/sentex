"""Backward-compatibility shim — imports from sentex.ingestion.llm."""
from .ingestion.llm import *  # noqa: F401, F403
from .ingestion.llm import generate_l0, generate_l2
