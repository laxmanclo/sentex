"""AgentContext — what each agent function receives when it runs."""
from __future__ import annotations

from typing import Any, Callable, Awaitable

from .types import AssembledContext


class AgentContext:
    """
    Passed to every agent function.

    Usage inside an agent:

        async def my_agent(ctx: AgentContext):
            sentences = ctx["search-results"]   # assembled sentences / text
            prompt = ctx.render()               # formatted prompt block
            answer = await ctx.llm(prompt)      # call the pipeline LLM
            return answer                       # ingested as the write node
    """

    def __init__(
        self,
        query: str,
        assembled: AssembledContext,
        llm_fn: Callable[[str], Awaitable[str]],
    ) -> None:
        self.query = query
        self.assembled = assembled
        self._llm_fn = llm_fn

    # ------------------------------------------------------------------
    # Context access
    # ------------------------------------------------------------------

    def __getitem__(self, node_id: str) -> str | list[str]:
        return self.assembled.context[node_id]

    def get(self, node_id: str, default: Any = None) -> Any:
        return self.assembled.context.get(node_id, default)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.assembled.context

    @property
    def context(self) -> dict[str, str | list[str]]:
        return self.assembled.context

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def render(self, separator: str = "\n\n") -> str:
        """Format assembled context into a prompt-ready string block.

        Example output:

            === search-results [L1 | 8 sentences | confidence 0.84] ===
            T-cells recognize antigens on pathogen surfaces.
            Antigen recognition triggers the adaptive immune cascade.
            ...

            === query [L3] ===
            Explain how the immune system works.
        """
        parts: list[str] = []
        for node_id, content in self.assembled.context.items():
            layer = self.assembled.layers_used.get(node_id, "?")
            conf = self.assembled.confidence.get(node_id)

            if isinstance(content, list):
                meta = f"L1 | {len(content)} sentences"
                if conf is not None:
                    meta += f" | confidence {conf:.2f}"
                header = f"=== {node_id} [{meta}] ==="
                body = "\n".join(content)
            else:
                meta = layer.upper()
                header = f"=== {node_id} [{meta}] ==="
                body = content

            parts.append(f"{header}\n{body}")

        return separator.join(parts)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def llm(self, prompt: str) -> str:
        """Call the pipeline LLM with *prompt*. Returns the response string."""
        return await self._llm_fn(prompt)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def token_utilization(self) -> float:
        return self.assembled.utilization

    @property
    def missing(self) -> list[str]:
        return self.assembled.missing

    @property
    def compressed(self) -> list[str]:
        return self.assembled.compressed

    def __repr__(self) -> str:
        return (
            f"AgentContext(nodes={list(self.assembled.context)}, "
            f"tokens={self.assembled.token_count}/{self.assembled.budget})"
        )
