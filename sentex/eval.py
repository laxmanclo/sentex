"""Backward-compatibility shim — imports from sentex.retrieval.eval."""
from .retrieval.eval import *  # noqa: F401, F403
from .retrieval.eval import RetrievalEvaluator, EvalCase, EvalResult, _ndcg, dataset_from_store
