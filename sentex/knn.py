"""Backward-compatibility shim — imports from sentex.ingestion.knn."""
from .ingestion.knn import *  # noqa: F401, F403
from .ingestion.knn import Adjacency, build_knn, update_knn
