"""Backward-compatibility shim — imports from sentex.pipeline.*"""
from .pipeline.pipeline import Pipeline, PipelineResult, AgentResult, _extractive_summary, _scope
from .pipeline.manifest import defineAgent
from .pipeline.context import AgentContext
from .pipeline.session import SessionRecord
