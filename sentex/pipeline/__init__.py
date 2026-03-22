"""Pipeline orchestration, agent context, and session tracking."""
from .pipeline import Pipeline, PipelineResult, AgentResult, _extractive_summary, _scope
from .context import AgentContext
from .manifest import defineAgent
from .session import SessionRecord
