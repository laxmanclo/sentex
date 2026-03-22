"""Pipeline — the main developer-facing SDK object.

Usage:

    from sentex import Pipeline, Read

    pipeline = Pipeline(persist="./sentex.db")  # optional cross-run memory

    @pipeline.agent(id="researcher", writes=["resources/search"])
    async def researcher(ctx):
        return web_search(ctx.query)

    @pipeline.agent(
        id="writer",
        reads=[Read("resources/search", layer="l1", budget=2000)],
        writes=["working/script"],
        token_budget=4000,
    )
    async def writer(ctx):
        return await ctx.llm(ctx.render())

    result = await pipeline.run(
        query="explain immune systems",
        llm="gpt-4o",
    )
    print(result.outputs["working/script"])
    print(result.summary())

---

Scopes (mirrors OpenViking's namespace model):
  resources/   → shared knowledge: search results, docs, reference data
  working/     → ephemeral computation: agent outputs within this run
  memory/      → extracted cross-run learnings (read from MemoryStore)
"""
from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .context import AgentContext
from ..core.graph import ContextGraph
from .manifest import defineAgent
from ..core.types import AgentManifest, AssembledContext, Read, Write


# ------------------------------------------------------------------
# LLM adapter
# ------------------------------------------------------------------

async def _litellm_call(model: str, prompt: str) -> str:
    import litellm
    resp = await asyncio.to_thread(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _make_llm_fn(llm: str | Callable) -> Callable[[str], Awaitable[str]]:
    if callable(llm):
        if asyncio.iscoroutinefunction(llm):
            return llm
        async def _wrap(prompt: str) -> str:
            return await asyncio.to_thread(llm, prompt)
        return _wrap
    async def _litellm(prompt: str) -> str:
        return await _litellm_call(llm, prompt)
    return _litellm


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class AgentResult:
    agent_id: str
    output: Any
    token_count: int
    budget: int
    utilization: float
    layers_used: dict[str, str]
    compressed: list[str]
    missing: list[str]
    confidence: dict[str, float]
    duration_ms: float


@dataclass
class PipelineResult:
    query: str
    session_id: str
    outputs: dict[str, Any]
    agent_results: list[AgentResult]
    graph: ContextGraph

    def summary(self) -> str:
        lines = [
            f"Session {self.session_id}",
            f"Query: {self.query!r}",
            "",
        ]
        for r in self.agent_results:
            filled = int(r.utilization * 20)
            bar = "█" * filled + "░" * (20 - filled)
            lines.append(
                f"  [{r.agent_id}]  {bar}  {r.token_count}/{r.budget} tok  {r.duration_ms:.0f}ms"
            )
            if r.layers_used:
                layers = "  ".join(f"{k}→{v.upper()}" for k, v in r.layers_used.items())
                lines.append(f"           {layers}")
            if r.confidence:
                conf = "  ".join(f"{k}: {v:.2f}" for k, v in r.confidence.items())
                lines.append(f"           conf: {conf}")
            if r.compressed:
                lines.append(f"           ⬇ compressed: {r.compressed}")
            if r.missing:
                lines.append(f"           ✗ missing:    {r.missing}")
        lines += [
            "",
            f"  graph: {self.graph.node_count} nodes · {self.graph.sentence_count} sentences",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------
# Registered agent entry
# ------------------------------------------------------------------

@dataclass
class _AgentEntry:
    manifest: AgentManifest
    fn: Callable


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class Pipeline:
    """
    Orchestrates a sequence of agents sharing a ContextGraph.

    Args:
        persist:  Path to a SQLite file for cross-run memory.
                  Edge weights and node summaries persist between runs.
                  Each run improves retrieval for the next one.
        graph:    Provide a pre-built ContextGraph (for testing / advanced use).
    """

    def __init__(
        self,
        graph: ContextGraph | None = None,
        persist: str | None = None,
    ) -> None:
        self.graph = graph or ContextGraph()
        self._agents: list[_AgentEntry] = []
        self._store = None
        if persist:
            from ..storage.store import MemoryStore
            self._store = MemoryStore(persist)
            self._apply_stored_boosts()

    def _apply_stored_boosts(self) -> None:
        if not self._store:
            return
        boosts = self._store.load_edge_boosts()
        # DB sentence IDs are 1-based; graph indices are 0-based
        for (src, dst), weight in boosts.items():
            self.graph._usage_boost[(src - 1, dst - 1)] = weight

    def agent(
        self,
        id: str,
        reads: list[Read] | None = None,
        writes: list[str | Write] | None = None,
        token_budget: int = 4000,
        fallback: str = "l2",
        confidence_threshold: float = 0.5,
    ) -> Callable:
        """Decorator that registers an agent with its manifest."""
        reads = reads or []
        writes = writes or []
        manifest = defineAgent(
            id=id,
            reads=reads,
            writes=writes,
            token_budget=token_budget,
            fallback=fallback,
            confidence_threshold=confidence_threshold,
        )
        def decorator(fn: Callable) -> Callable:
            self._agents.append(_AgentEntry(manifest=manifest, fn=fn))
            return fn
        return decorator

    def validate(self) -> list[str]:
        return self.graph.validate_pipeline([e.manifest for e in self._agents])

    async def run(
        self,
        query: str,
        llm: str | Callable = "gpt-4o",
        seed_context: dict[str, Any] | None = None,
        generate_summaries: bool = False,
        session_id: str | None = None,
    ) -> PipelineResult:
        """
        Execute all agents in registration order.

        Args:
            query:             The task driving the pipeline.
            llm:               LiteLLM model string OR any sync/async callable.
            seed_context:      {node_id: content} to pre-load before agents run.
                               Tip: use scope prefixes like "resources/query".
            generate_summaries: Generate L0/L2 via LLM at ingestion time.
                               Off by default for fast dev loops.
            session_id:        Name this run. Auto-generated if omitted.
        """
        session_id = session_id or f"run-{uuid.uuid4().hex[:8]}"

        # 0. Pre-load memory/ nodes from persistent store (cross-run learnings)
        if self._store:
            memory = self._store.read_memory("memory")
            for key, value in memory.items():
                self.graph.ingest(
                    f"memory/{key}", value, agent_id="__memory__",
                    generate_summaries=False,
                )

        # 1. Seed context
        if seed_context:
            for node_id, content in seed_context.items():
                cached = self._store.load_node_summary(
                    _scope(node_id), node_id
                ) if self._store else None

                node = self.graph.ingest(
                    node_id, content, agent_id="__seed__",
                    generate_summaries=generate_summaries,
                )
                if cached:
                    node.l0 = node.l0 or cached["l0"]
                    node.l2 = node.l2 or cached["l2"]
                if not node.l2:
                    node.l2 = _extractive_summary(
                        content if isinstance(content, str) else str(content)
                    )

        # 2. Validate
        errors = self.validate()
        if errors:
            raise ValueError("Pipeline validation failed:\n" + "\n".join(errors))

        llm_fn = _make_llm_fn(llm)
        outputs: dict[str, Any] = {}
        agent_results: list[AgentResult] = []

        if self._store:
            self._store.record_session(
                session_id, query,
                [e.manifest.id for e in self._agents], [],
            )

        # 3. Run agents in order
        for entry in self._agents:
            manifest = entry.manifest
            t0 = time.perf_counter()

            assembled = self.graph.assemble_for(manifest, query)
            ctx = AgentContext(query=query, assembled=assembled, llm_fn=llm_fn)

            result = entry.fn(ctx)
            if asyncio.iscoroutine(result) or inspect.isawaitable(result):
                result = await result

            # Ingest each declared write node
            for write in manifest.writes:
                node_id = write.node_id if isinstance(write, Write) else write
                if len(manifest.writes) == 1:
                    content = result
                else:
                    content = result.get(node_id, "") if isinstance(result, dict) else str(result)

                content_str = content if isinstance(content, str) else str(content)

                cached = self._store.load_node_summary(
                    _scope(node_id), node_id
                ) if self._store else None

                node = self.graph.ingest(
                    node_id, content_str, agent_id=manifest.id,
                    generate_summaries=generate_summaries,
                )
                if cached:
                    node.l0 = node.l0 or cached["l0"]
                    node.l2 = node.l2 or cached["l2"]

                # Extractive L2 fallback — never let l2 be an empty string.
                # The fallback chain l1→l2→l0 breaks if l2 is "".
                if not node.l2:
                    node.l2 = _extractive_summary(content_str)

                outputs[node_id] = content

                if self._store:
                    self._store.save_node_summary(
                        _scope(node_id), node_id,
                        node.l0, node.l2, content_str, manifest.id,
                    )

            # Track usage → boost edge weights in-graph and in store
            used_ids = [
                r.node_id for r in manifest.reads
                if hasattr(r, "node_id") and r.node_id in assembled.context
            ]
            # Also mark auto: keys as used
            auto_used = [k[5:] for k in assembled.context if k.startswith("auto:")]
            used_ids.extend(auto_used)
            self.graph.mark_used(assembled, used_ids)

            if self._store and used_ids:
                for nid in used_ids:
                    n = self.graph.get_node(nid)
                    if n:
                        for i in n.sentence_ids:
                            for j, _ in self.graph._adjacency.get(i, []):
                                self._store.boost_edge(i + 1, j + 1)
                self._store.flush_edge_boosts()

            duration_ms = (time.perf_counter() - t0) * 1000
            agent_results.append(AgentResult(
                agent_id=manifest.id,
                output=result,
                token_count=assembled.token_count,
                budget=assembled.budget,
                utilization=assembled.utilization,
                layers_used=assembled.layers_used,
                compressed=assembled.compressed,
                missing=assembled.missing,
                confidence=assembled.confidence,
                duration_ms=duration_ms,
            ))

        if self._store:
            self._store.commit_session(session_id)

        return PipelineResult(
            query=query,
            session_id=session_id,
            outputs=outputs,
            agent_results=agent_results,
            graph=self.graph,
        )

    def history(self) -> list[dict]:
        """Return all past sessions recorded to the MemoryStore."""
        if not self._store:
            return []
        return self._store.all_sessions()

    def save(self, path: str) -> None:
        from . import persistence
        persistence.save(self.graph, path)

    def load(self, path: str) -> None:
        from . import persistence
        persistence.load(path, self.graph)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _scope(node_id: str) -> str:
    return node_id.split("/")[0] if "/" in node_id else "working"


def _extractive_summary(text: str, max_tokens: int = 400) -> str:
    """First-N-tokens extractive summary. Used as L2 fallback when LLM is off."""
    from ..ingestion.splitter import split_sentences
    from ..core.tokens import count_tokens

    sentences = split_sentences(text)
    collected, total = [], 0
    for s in sentences:
        t = count_tokens(s)
        if total + t > max_tokens:
            break
        collected.append(s)
        total += t
    return " ".join(collected) if collected else text[:600]
