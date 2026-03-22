"""Backward-compatibility shim — imports from sentex.storage.persistence."""
from .storage.persistence import *  # noqa: F401, F403
from .storage.persistence import load, save
