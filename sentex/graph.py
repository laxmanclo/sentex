"""ContextGraph — the central in-memory store.

Holds:
- global sentence array (embeddings + text + metadata)
- per-node ContextNode records
- KNN adjacency dict
- usage weights (edge boosts)
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from .embedder import Embedder
from .knn import Adjacency, build_knn, update_knn
from .llm import generate_l0, generate_l2
from .splitter import split_sentences
from .tokens import count_tokens, count_tokens_list
from .types import (
    AgentManifest,
    AssembledContext,
    AutoRead,
    ContextNode,
    L0L1L2L3TokenCounts,
    Read,
    SentenceMetadata,
)
from .retrieval import retrieve_l1


# Fallback order when content is too large for budget: biggest → smallest.
# L1 is placed AFTER L2 because L2 is a fixed ~2k summary whereas L1 is
# budget-controlled — when total agent budget is tight, L2 may still be too
# large and we fall to L1 (with a tighter budget) then L0.
_LAYER_ORDER = ["l3", "l2", "l1", "l0"]

# Fallback when L1 is declared but total-budget check fires: try L2 first
# (the designed narrative fallback), then L0.
_L1_BUDGET_FALLBACK = ["l2", "l0"]


class ContextGraph:
    def __init__(
        self,
        embedder: Embedder | None = None,
        knn_k: int = 5,
        llm_model: str | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.knn_k = knn_k
        self.llm_model = llm_model

        # Global sentence store — shape (0,) until first ingest sets actual dim
        self._embeddings: np.ndarray = np.empty((0,), dtype=np.float32)
        self._sentences: list[str] = []
        self._metadata: list[SentenceMetadata] = []

        # KNN adjacency with usage-boosted weights
        self._adjacency: Adjacency = {}
        self._usage_boost: dict[tuple[int, int], float] = {}  # (i,j) → extra weight

        # Context nodes
        self._nodes: dict[str, ContextNode] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        node_id: str,
        content: Any,
        agent_id: str,
        generate_summaries: bool = False,
    ) -> ContextNode:
        """Ingest *content* as a new ContextNode.

        Splits into sentences, embeds, updates KNN, stores extractive L0/L2.

        Args:
            generate_summaries: If True, fires async LLM calls to generate
                high-quality L0 identity and L2 summary (requires ENGRAM_LLM_MODEL
                or OPENAI_API_KEY). Defaults to False — extractive fallbacks are
                always built from the content itself and work without an LLM.
        """
        text = content if isinstance(content, str) else str(content)
        sentences = split_sentences(text)

        start_idx = len(self._sentences)

        # Embed new sentences
        if sentences:
            vecs = self.embedder.embed(sentences)
            # On first ingest, _embeddings is 1-D placeholder — replace directly.
            # On subsequent ingests, vstack (both arrays now have shape (n, D)).
            if self._embeddings.ndim == 1:
                self._embeddings = vecs
            else:
                self._embeddings = np.vstack([self._embeddings, vecs])
            self._sentences.extend(sentences)
            for i, s in enumerate(sentences):
                self._metadata.append(
                    SentenceMetadata(
                        sentence_idx=start_idx + i,
                        node_id=node_id,
                        agent_id=agent_id,
                    )
                )

        # Rebuild / update KNN
        if len(self._sentences) > 0:
            if start_idx == 0:
                self._adjacency = build_knn(self._embeddings, k=self.knn_k)
            else:
                self._adjacency = update_knn(
                    self._embeddings, self._adjacency, start_idx, k=self.knn_k
                )

        sentence_ids = list(range(start_idx, start_idx + len(sentences)))

        l1_tokens = count_tokens_list(sentences)
        l3_tokens = count_tokens(text)

        node = ContextNode(
            id=node_id,
            produced_by=agent_id,
            l3=content,
            sentence_ids=sentence_ids,
            first_sentence=sentences[0] if sentences else "",
            token_counts=L0L1L2L3TokenCounts(l1=l1_tokens, l3=l3_tokens),
        )
        # Always build extractive L2 from the first N sentences so the
        # L1→L2 fallback chain never hits an empty string, even without LLM.
        node.l2 = _build_extractive_l2(sentences)
        node.token_counts.l2 = count_tokens(node.l2)

        self._nodes[node_id] = node

        if generate_summaries and sentences:
            asyncio.ensure_future(self._generate_summaries(node_id, text))

        return node

    async def ingest_async(
        self,
        node_id: str,
        content: Any,
        agent_id: str,
    ) -> ContextNode:
        """Ingest and await summary generation."""
        node = self.ingest(node_id, content, agent_id, generate_summaries=False)
        text = content if isinstance(content, str) else str(content)
        await self._generate_summaries(node_id, text)
        return node

    async def _generate_summaries(self, node_id: str, text: str) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            return
        l0, l2 = await asyncio.gather(
            generate_l0(text, model=self.llm_model),
            generate_l2(text, model=self.llm_model),
        )
        node.l0 = l0
        node.l2 = l2
        node.token_counts.l0 = count_tokens(l0)
        node.token_counts.l2 = count_tokens(l2)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _effective_adjacency(self) -> Adjacency:
        """Return adjacency with usage boosts applied."""
        if not self._usage_boost:
            return self._adjacency
        boosted: Adjacency = {}
        for i, neighbors in self._adjacency.items():
            new_neighbors = [
                (j, sim + self._usage_boost.get((i, j), 0.0))
                for j, sim in neighbors
            ]
            new_neighbors.sort(key=lambda x: -x[1])
            boosted[i] = new_neighbors
        return boosted

    def retrieve(
        self,
        node_id: str,
        layer: str,
        query: str,
        budget_tokens: int,
        confidence_threshold: float = 0.5,
        fallback: str = "l2",
    ) -> tuple[str | list[str], str, float]:
        """Return (content, layer_used, confidence).

        Handles layer fallback automatically. layer_used in the return value
        always reflects what was actually served, not what was declared.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return "", layer, 0.0

        if layer == "l3":
            txt = node.l3 if isinstance(node.l3, str) else str(node.l3)
            return txt, "l3", 1.0

        if layer == "l2":
            content = node.l2 or _extractive_l2(node) or node.l0
            actual = "l2" if node.l2 else ("l0" if not node.l2 and node.l0 else "l2")
            return content, actual, 1.0

        if layer == "l0":
            content = node.l0 or _extractive_l0(node)
            return content, "l0", 1.0

        # layer == "l1"
        if not node.sentence_ids or len(self._embeddings) == 0:
            # No sentences ingested yet — serve L2 fallback
            content = node.l2 or _extractive_l2(node) or node.l0
            return content, fallback, 0.0

        query_vec = self.embedder.embed_one(query)
        adj = self._effective_adjacency()
        sentences, confidence = retrieve_l1(
            query_vec,
            self._embeddings,
            self._sentences,
            adj,
            budget_tokens,
            candidate_ids=node.sentence_ids,
        )

        if confidence < confidence_threshold:
            content = node.l2 or _extractive_l2(node) or node.l0
            actual = "l2" if (node.l2 or _extractive_l2(node)) else "l0"
            return content, actual, confidence

        return sentences, "l1", confidence

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble_for(
        self,
        agent: AgentManifest,
        query: str,
    ) -> AssembledContext:
        context: dict[str, str | list[str]] = {}
        layers_used: dict[str, str] = {}
        compressed: list[str] = []
        missing: list[str] = []
        confidence: dict[str, float] = {}

        token_total = 0

        for read in agent.reads:
            # AutoRead: scan nodes at L0, retrieve top-k dynamically
            if isinstance(read, AutoRead):
                results = self.retrieve_auto(
                    query=query,
                    top_k=read.top_k,
                    layer=read.layer,
                    budget_per_node=read.budget_per_node,
                    scope=read.scope,
                    confidence_threshold=agent.confidence_threshold,
                    fallback=agent.fallback,
                )
                for node_id, content, layer_used, conf in results:
                    key = f"auto:{node_id}"
                    content_tokens = (
                        count_tokens_list(content)
                        if isinstance(content, list)
                        else count_tokens(content)
                    )
                    if token_total + content_tokens <= agent.token_budget:
                        context[key] = content
                        layers_used[key] = layer_used
                        confidence[key] = conf
                        token_total += content_tokens
                continue
            node_id = read.node_id
            if node_id not in self._nodes:
                missing.append(node_id)
                continue

            content, layer_used, conf = self.retrieve(
                node_id=node_id,
                layer=read.layer,
                query=query,
                budget_tokens=read.budget,
                confidence_threshold=agent.confidence_threshold,
                fallback=agent.fallback,
            )

            # Budget enforcement against total agent budget
            content_tokens = (
                count_tokens_list(content)
                if isinstance(content, list)
                else count_tokens(content)
            )

            remaining = agent.token_budget - token_total
            if content_tokens > remaining:
                # Choose fallback order: L1 falls back to L2 then L0 (not skip L2).
                if layer_used == "l1":
                    fallback_order = _L1_BUDGET_FALLBACK
                else:
                    fallback_order = _LAYER_ORDER[_LAYER_ORDER.index(layer_used) + 1:]
                for fb_layer in fallback_order:
                    fb_content, fb_layer_used, fb_conf = self.retrieve(
                        node_id=node_id,
                        layer=fb_layer,
                        query=query,
                        budget_tokens=remaining,
                        confidence_threshold=agent.confidence_threshold,
                        fallback=agent.fallback,
                    )
                    fb_tokens = (
                        count_tokens_list(fb_content)
                        if isinstance(fb_content, list)
                        else count_tokens(fb_content)
                    )
                    if fb_tokens <= remaining:
                        content, layer_used, conf = fb_content, fb_layer_used, fb_conf
                        content_tokens = fb_tokens
                        compressed.append(node_id)
                        break
                else:
                    # Nothing fits — serve the shortest possible representation
                    node = self._nodes[node_id]
                    l0_text = node.l0 or _extractive_l0(node)
                    # Token-aware truncation: encode, slice, decode
                    content = _truncate_to_tokens(l0_text, remaining)
                    layer_used = "l0"
                    content_tokens = count_tokens(content)
                    compressed.append(node_id)

            context[node_id] = content
            layers_used[node_id] = layer_used
            confidence[node_id] = conf
            token_total += content_tokens

        utilization = token_total / agent.token_budget if agent.token_budget else 0.0

        return AssembledContext(
            context=context,
            token_count=token_total,
            budget=agent.token_budget,
            utilization=utilization,
            layers_used=layers_used,
            compressed=compressed,
            missing=missing,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def mark_used(
        self,
        assembled: AssembledContext,
        used_ids: list[str],
        boost: float = 0.05,
    ) -> None:
        """Boost edge weights for sentences in nodes that were marked useful.

        Boosts BOTH intra-node edges (sentences within the same node) and
        cross-node edges (sentences in used nodes connected to sentences in
        other used nodes). Cross-node boosting is what reinforces the
        cross-agent graph connections over time.
        """
        # Collect all sentence IDs across all used nodes
        all_used_sentence_ids: set[int] = set()
        for node_id in used_ids:
            node = self._nodes.get(node_id)
            if node:
                all_used_sentence_ids.update(node.sentence_ids)

        # Boost every edge where the source sentence is in a used node.
        # Destination can be in any node — this is the cross-node boost.
        for i in all_used_sentence_ids:
            for j, _ in self._adjacency.get(i, []):
                if j in all_used_sentence_ids:
                    # Stronger boost for edges within the used set
                    self._usage_boost[(i, j)] = (
                        self._usage_boost.get((i, j), 0.0) + boost
                    )
                else:
                    # Weaker boost for edges pointing outside used set
                    self._usage_boost[(i, j)] = (
                        self._usage_boost.get((i, j), 0.0) + boost * 0.3
                    )

    # ------------------------------------------------------------------
    # Node-level retrieval (L0 scan)
    # ------------------------------------------------------------------

    def scan_nodes(
        self,
        query: str,
        top_k: int = 5,
        scope: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find the top-k most relevant nodes by embedding their L0 summaries.

        This is node-level retrieval — answering "which nodes are relevant to
        this query?" before doing sentence-level (L1) retrieval inside them.
        Equivalent to OpenViking's directory-level abstract scan.

        Returns list of (node_id, similarity_score) sorted descending.

        Args:
            query:  The query to match against.
            top_k:  How many nodes to return.
            scope:  Optional scope prefix filter, e.g. "resources" to only
                    scan nodes whose ID starts with "resources/".
        """
        if not self._nodes:
            return []

        query_vec = self.embedder.embed_one(query)

        candidates: list[tuple[str, float]] = []
        for node_id, node in self._nodes.items():
            if scope and not node_id.startswith(scope):
                continue

            # Use L0 if available; fall back to extractive L0 from L2/sentences
            l0_text = node.l0 or _extractive_l0(node)
            if not l0_text:
                continue

            l0_vec = self.embedder.embed_one(l0_text)
            score = float(np.dot(query_vec, l0_vec))
            candidates.append((node_id, score))

        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def retrieve_auto(
        self,
        query: str,
        top_k: int = 3,
        layer: str = "l1",
        budget_per_node: int = 1000,
        scope: str | None = None,
        confidence_threshold: float = 0.5,
        fallback: str = "l2",
    ) -> list[tuple[str, str | list[str], str, float]]:
        """Scan nodes at L0, then retrieve from top-k at the declared layer.

        Returns list of (node_id, content, layer_used, confidence).

        This is the dynamic equivalent of declaring a Read — useful when you
        don't know the node IDs at pipeline-definition time.
        """
        top_nodes = self.scan_nodes(query, top_k=top_k, scope=scope)
        results = []
        for node_id, _score in top_nodes:
            content, layer_used, conf = self.retrieve(
                node_id=node_id,
                layer=layer,
                query=query,
                budget_tokens=budget_per_node,
                confidence_threshold=confidence_threshold,
                fallback=fallback,
            )
            results.append((node_id, content, layer_used, conf))
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_pipeline(self, agents: list[AgentManifest]) -> list[str]:
        """Check that every declared read will be produced before the agent runs.

        Returns list of error strings (empty = valid).
        """
        errors: list[str] = []
        produced: set[str] = set(self._nodes.keys())
        for agent in agents:
            for read in agent.reads:
                if isinstance(read, AutoRead):
                    continue  # AutoRead discovers nodes dynamically — nothing to validate
                if read.node_id not in produced:
                    errors.append(
                        f"Agent '{agent.id}' reads '{read.node_id}' "
                        f"which is not produced before it runs."
                    )
            for write in agent.writes:
                produced.add(write.node_id)
        return errors

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> ContextNode | None:
        return self._nodes.get(node_id)

    @property
    def sentence_count(self) -> int:
        return len(self._sentences)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Simple interface for bring-your-own-agent usage
    # ------------------------------------------------------------------

    def put(self, node_id: str, content: str, agent_id: str = "agent") -> "ContextNode":
        """Store an agent's output. Alias for ingest() with a simpler signature.

        Call this after your agent produces output:
            graph.put("search-results", my_agent.run(task))
        """
        return self.ingest(node_id, content, agent_id=agent_id, generate_summaries=False)

    def get(
        self,
        node_id: str,
        query: str,
        budget: int = 2000,
        layer: str = "l1",
    ) -> str | list[str]:
        """Retrieve context for a node. Returns sentences (L1) or text (L0/L2/L3).

        Call this to build context for your next agent:
            context = graph.get("search-results", query="write a script", budget=2000)
            prompt = f"Use this:\\n{context}\\n\\nNow write the script."

        Returns:
            list[str] for layer="l1" (sentences, ordered by relevance)
            str       for layer="l0" / "l2" / "l3"
        """
        content, _layer_used, _conf = self.retrieve(
            node_id=node_id,
            layer=layer,
            query=query,
            budget_tokens=budget,
        )
        return content

    def render(self, node_id: str, query: str, budget: int = 2000) -> str:
        """Get context as a single formatted string, ready to drop into a prompt.

        Retrieves at L1 (sentence graph), formats as joined text.

            prompt = f"Context:\\n{graph.render('search-results', query)}\\n\\nTask: ..."
        """
        content = self.get(node_id, query=query, budget=budget, layer="l1")
        if isinstance(content, list):
            return "\n".join(content)
        return content

    def used(self, node_id: str) -> None:
        """Mark a node as used after your agent ran. Boosts future retrieval.

        Call after the agent that consumed this node completes:
            graph.used("search-results")
        """
        from .types import AssembledContext
        dummy = AssembledContext(
            context={node_id: []}, token_count=0, budget=0, utilization=0.0,
            layers_used={}, compressed=[], missing=[], confidence={},
        )
        self.mark_used(dummy, used_ids=[node_id])


# ------------------------------------------------------------------
# Module-level helpers (used by retrieve and scan)
# ------------------------------------------------------------------

def _extractive_l0(node: "ContextNode") -> str:
    """Extractive L0: first sentence stored at ingest time, or first sentence of L2."""
    if node.first_sentence:
        return node.first_sentence
    if node.l2:
        first = node.l2.split(".")[0].strip()
        return (first + ".") if first and not first.endswith(".") else first
    return ""


def _extractive_l2(node: "ContextNode") -> str:
    """Return node.l2 if non-empty, else empty string. Explicit for clarity."""
    return node.l2 or ""


def _build_extractive_l2(sentences: list[str], max_tokens: int = 300) -> str:
    """Build an extractive L2 from the first N sentences up to max_tokens."""
    from .tokens import count_tokens as _ct
    collected, total = [], 0
    for s in sentences:
        t = _ct(s)
        if total + t > max_tokens:
            break
        collected.append(s)
        total += t
    return " ".join(collected)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens using the actual tokenizer."""
    from .tokens import _get_encoder
    if max_tokens <= 0 or not text:
        return ""
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
