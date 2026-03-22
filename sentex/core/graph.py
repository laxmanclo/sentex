"""ContextGraph — the central in-memory store.

Holds:
- global sentence array (embeddings + text + metadata)
- per-node ContextNode records
- KNN adjacency dict
- usage weights with hotness scoring (frequency × recency decay)
- bidirectional node relations
- optional telemetry collection
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from ..ingestion.embedder import Embedder
from ..ingestion.knn import Adjacency, build_knn, update_knn
from ..ingestion.llm import generate_l0, generate_l2
from .relations import RelationIndex, RelationKind
from .scoring import HotnessScore, compute_hotness
from ..ingestion.splitter import split_sentences
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
from ..retrieval.engine import retrieve_l1


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
        embedder: "Embedder | None" = None,
        knn_k: int = 5,
        llm_model: str | None = None,
        metrics: "Any | None" = None,
        hotness_freq_scale: float = 10.0,
        hotness_half_life_h: float = 24.0,
        confidence_mode: str = "adaptive",
        cross_node_k: int = 1,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.knn_k = knn_k
        self.llm_model = llm_model
        self._metrics = metrics  # MetricsCollector | None

        # Hotness scoring parameters
        self._hotness_freq_scale = hotness_freq_scale
        self._hotness_half_life_s = hotness_half_life_h * 3600

        # Confidence mode: "adaptive" (relative to corpus) or "absolute" (fixed threshold)
        # Adaptive is better for mixed-domain pipelines; absolute is simpler and predictable.
        self._confidence_mode = confidence_mode

        # After KNN, guarantee this many cross-node edges per node-pair.
        # Prevents cross-agent edges being crowded out by intra-node edges at small k.
        self._cross_node_k = cross_node_k

        # Global sentence store — shape (0,) until first ingest sets actual dim
        self._embeddings: np.ndarray = np.empty((0,), dtype=np.float32)
        self._sentences: list[str] = []
        self._metadata: list[SentenceMetadata] = []

        # KNN adjacency + hotness-scored edge boosts
        self._adjacency: Adjacency = {}
        self._usage_boost: dict[tuple[int, int], HotnessScore] = {}

        # Context nodes
        self._nodes: dict[str, ContextNode] = {}

        # Bidirectional node relations
        self._relations: RelationIndex = RelationIndex()

        # Lazy filesystem view
        self._fs: Any = None

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
            # Guarantee cross-node edges so BFS can always traverse agent boundaries.
            # Without this, small k can crowd out cross-node edges with intra-node ones.
            if self._cross_node_k > 0 and len(self._nodes) >= 2:
                _ensure_cross_node_edges(
                    self._embeddings, self._adjacency, self._nodes, k=self._cross_node_k
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
        # Build centroid-based extractive L2: pick the most *representative*
        # sentences (closest to the document centroid) rather than the first N.
        # This means mid-document content surfaces in L2 — critical for the
        # L1-confidence-fails → L2-fallback path to remain useful.
        node.l2 = _build_centroid_l2(sentences, vecs if sentences else None)
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
        """Return adjacency with hotness-scored boosts applied.

        Hotness = sigmoid(hit_count / freq_scale) × exp(-ln2 × age / half_life)
        This replaces the old flat additive boost with a principled model that
        decays over time and saturates with frequency.
        """
        if not self._usage_boost:
            return self._adjacency
        now = time.time()
        boosted: Adjacency = {}
        for i, neighbors in self._adjacency.items():
            new_neighbors = []
            for j, sim in neighbors:
                score = self._usage_boost.get((i, j))
                extra = (
                    compute_hotness(
                        score, now=now,
                        freq_scale=self._hotness_freq_scale,
                        half_life_s=self._hotness_half_life_s,
                    )
                    if score is not None
                    else 0.0
                )
                new_neighbors.append((j, sim + extra))
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
    ) -> tuple[list[str], str, float]:
        """Return (content, layer_used, confidence).

        content is ALWAYS list[str] — single-element for L0/L2/L3, multi for L1.
        This eliminates the str | list[str] ambiguity that caused silent bugs
        when callers iterated over a fallback string character by character.

        Confidence mode (set on ContextGraph init):
          "adaptive" — threshold is relative to the query's similarity distribution
                       across this node's sentences. Works well for mixed-domain
                       pipelines where all cosines are low.
          "absolute" — threshold is the raw confidence_threshold float (old behaviour).
        """
        node = self._nodes.get(node_id)
        if node is None:
            return [""], layer, 0.0

        if layer == "l3":
            txt = node.l3 if isinstance(node.l3, str) else str(node.l3)
            return [txt], "l3", 1.0

        if layer == "l2":
            content = node.l2 or _extractive_l2(node) or node.l0
            actual = "l2" if node.l2 else ("l0" if node.l0 else "l2")
            return [content], actual, 1.0

        if layer == "l0":
            content = node.l0 or _extractive_l0(node)
            return [content], "l0", 1.0

        # layer == "l1"
        if not node.sentence_ids or len(self._embeddings) == 0:
            content = node.l2 or _extractive_l2(node) or node.l0
            return [content] if content else [""], fallback, 0.0

        query_vec = self.embedder.embed_one(query)
        adj = self._effective_adjacency()
        sentences, confidence, converged = retrieve_l1(
            query_vec,
            self._embeddings,
            self._sentences,
            adj,
            budget_tokens,
            candidate_ids=node.sentence_ids,
        )

        # Adaptive threshold: check if confidence is meaningfully above the
        # median similarity for this node. In mixed-domain corpora all cosines
        # are low — a fixed 0.5 would always fall back to L2 truncation.
        if self._confidence_mode == "adaptive":
            fires = _l1_fires_adaptive(
                query_vec, self._embeddings, node.sentence_ids, confidence
            )
        else:
            fires = confidence >= confidence_threshold

        # Record telemetry
        if self._metrics is not None:
            from ..telemetry.collector import OperationMetrics
            self._metrics.record(OperationMetrics(
                operation="retrieve",
                node_id=node_id,
                duration_ms=0.0,
                sentences_in=len(sentences),
                tokens_out=count_tokens_list(sentences),
                layer_used="l1" if fires else fallback,
                confidence=confidence,
                converged=converged,
            ))

        if not fires:
            content = node.l2 or _extractive_l2(node) or node.l0
            actual = "l2" if (node.l2 or _extractive_l2(node)) else "l0"
            return [content] if content else [""], actual, confidence

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
        # HotnessScore.hit() records timestamp for recency decay.
        now = time.time()
        for i in all_used_sentence_ids:
            for j, _ in self._adjacency.get(i, []):
                key = (i, j)
                if key not in self._usage_boost:
                    self._usage_boost[key] = HotnessScore()
                self._usage_boost[key].hit(now=now)

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

        scores: dict[str, float] = {}
        for node_id, node in self._nodes.items():
            if scope and not node_id.startswith(scope):
                continue

            # Use centroid-based L2 as L0 if no explicit L0 set — it's a better
            # document representative than just the first sentence.
            l0_text = node.l0 or node.l2 or _extractive_l0(node)
            if not l0_text:
                continue

            l0_vec = self.embedder.embed_one(l0_text)
            scores[node_id] = float(np.dot(query_vec, l0_vec))

        # Relation-based score propagation: if node B is derived_from / summarizes
        # node A, and A is relevant, boost B's score proportionally.
        # ALPHA=0.3 means a highly-relevant source contributes 0.3 × its score
        # to derived nodes. This makes explicit provenance chains useful for retrieval.
        _RELATION_ALPHA = 0.3
        _PROPAGATING_KINDS = {"derived_from", "summarizes", "extends"}
        for rel in self._relations.all_relations():
            if rel.kind not in _PROPAGATING_KINDS:
                continue
            src_score = scores.get(rel.src, 0.0)
            dst_score = scores.get(rel.dst, 0.0)
            if dst_score > 0 and rel.dst in scores:
                # dst is the source of knowledge; boost src (the derivative)
                scores[rel.src] = scores.get(rel.src, 0.0) + _RELATION_ALPHA * dst_score
            if src_score > 0 and rel.src in scores:
                # src derived from dst; also lightly boost dst (discovery direction)
                scores[rel.dst] = scores.get(rel.dst, 0.0) + _RELATION_ALPHA * 0.5 * src_score

        candidates = sorted(scores.items(), key=lambda x: -x[1])
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

    def stats(self) -> dict:
        """Return a summary of graph state — useful for debugging and monitoring."""
        total_edges = sum(len(v) for v in self._adjacency.values())
        hot_edges = sum(
            1 for s in self._usage_boost.values() if s.hit_count > 0
        )
        return {
            "nodes": self.node_count,
            "sentences": self.sentence_count,
            "edges": total_edges,
            "hot_edges": hot_edges,
            "relations": len(self._relations),
            "node_ids": list(self._nodes.keys()),
        }

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def link(
        self,
        src: str,
        dst: str,
        kind: RelationKind = "related",
        weight: float = 1.0,
        created_by: str = "__system__",
    ):
        """Create an explicit semantic relation from src → dst.

        Example:
            graph.link("working/analysis", "resources/search", kind="derived_from")
        """
        return self._relations.link(src, dst, kind=kind, weight=weight, created_by=created_by)

    def unlink(self, src: str, dst: str, kind: RelationKind | None = None) -> int:
        """Remove relations between src and dst. Returns count removed."""
        return self._relations.unlink(src, dst, kind=kind)

    def neighbors(
        self,
        node_id: str,
        direction: str = "both",
        kind: RelationKind | None = None,
    ) -> list:
        """Return Relation objects for a node's explicit relations."""
        return self._relations.neighbors(node_id, direction=direction, kind=kind)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Filesystem view
    # ------------------------------------------------------------------

    @property
    def fs(self):
        """Virtual filesystem view over this graph (ls, tree, stat, find).

        Example:
            graph.fs.ls("resources")
            graph.fs.tree()
            graph.fs.stat("resources/search")
            graph.fs.find("working/*analysis*")
        """
        if self._fs is None:
            from .fs import GraphFS
            self._fs = GraphFS(self)
        return self._fs

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
    ) -> list[str]:
        """Retrieve context for a node. Always returns list[str].

        For L1: multiple sentences ordered by relevance.
        For L0/L2/L3: single-element list containing the text.

        This consistent return type means `for sentence in graph.get(...)` is
        always safe — no more silent character iteration when fallback fires.

        Call this to build context for your next agent:
            sentences = graph.get("search-results", query="write a script", budget=2000)
            prompt = f"Use this:\\n{'\\n'.join(sentences)}\\n\\nNow write the script."
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

            prompt = f"Context:\\n{graph.render('search-results', query)}\\n\\nTask: ..."
        """
        return "\n".join(self.get(node_id, query=query, budget=budget, layer="l1"))

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


def _build_centroid_l2(
    sentences: list[str],
    vecs: "np.ndarray | None",
    max_tokens: int = 300,
) -> str:
    """Build an extractive L2 by selecting sentences closest to the document centroid.

    Unlike first-N truncation, this surfaces content from anywhere in the document —
    critical for the L1-confidence-low → L2-fallback path. A query about content in
    the middle of a long document now gets a representative summary, not just the header.

    Falls back to first-N if embeddings are unavailable.
    """
    from .tokens import count_tokens as _ct
    if not sentences:
        return ""
    if vecs is None or len(vecs) == 0:
        # Fallback: first-N (old behaviour, only when embeddings unavailable)
        collected, total = [], 0
        for s in sentences:
            t = _ct(s)
            if total + t > max_tokens:
                break
            collected.append(s)
            total += t
        return " ".join(collected)

    # Centroid = mean of all sentence embeddings, normalised
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm

    # Score each sentence by similarity to centroid
    sims = vecs @ centroid  # (n,)
    order = np.argsort(-sims)  # highest-sim first

    # Greedy selection within token budget — prefer high-sim sentences but
    # allow gaps so a long sentence doesn't block shorter but relevant ones.
    selected: list[tuple[int, str]] = []
    total = 0
    for idx in order:
        s = sentences[idx]
        t = _ct(s)
        if total + t > max_tokens:
            continue  # try next sentence (don't break — shorter ones may fit)
        selected.append((int(idx), s))
        total += t
        if total >= max_tokens:
            break

    # Re-sort by original position to maintain narrative flow
    selected.sort(key=lambda x: x[0])
    return " ".join(s for _, s in selected)


def _build_extractive_l2(sentences: list[str], max_tokens: int = 300) -> str:
    """First-N truncation fallback (kept for backward compat, prefer _build_centroid_l2)."""
    from .tokens import count_tokens as _ct
    collected, total = [], 0
    for s in sentences:
        t = _ct(s)
        if total + t > max_tokens:
            break
        collected.append(s)
        total += t
    return " ".join(collected)


def _l1_fires_adaptive(
    query_vec: "np.ndarray",
    embeddings: "np.ndarray",
    sentence_ids: list[int],
    confidence: float,
    z: float = 0.5,
) -> bool:
    """Return True if L1 retrieval is worth serving for this query.

    Instead of comparing confidence against a fixed 0.5, we compare it against
    the distribution of similarities for this node's sentences.

    Fires if: confidence >= median(node_sims) + z * std(node_sims)

    Why this works for mixed-domain corpora:
    - In a mixed pipeline (company + tech + market), all cosines are low (0.2-0.4)
    - Fixed threshold 0.5 → always falls back to L2 truncation
    - Adaptive: if median is 0.22 and std is 0.08, threshold = 0.26 → fires correctly

    z=0.5 means "entry point must be half a std above the node median".
    Higher z = more selective (fewer L1 firings), lower z = more permissive.
    """
    if len(sentence_ids) == 0:
        return False
    idx_arr = np.array(sentence_ids, dtype=np.int32)
    node_sims = embeddings[idx_arr] @ query_vec
    threshold = float(np.median(node_sims)) + z * float(np.std(node_sims))
    return confidence >= max(threshold, 0.05)  # floor to prevent degenerate cases


def _ensure_cross_node_edges(
    embeddings: "np.ndarray",
    adjacency: "Adjacency",
    nodes: "dict",
    k: int = 1,
) -> None:
    """Guarantee at least k cross-node KNN edges per node-pair.

    With small k (e.g. k=5) on coherent documents, all KNN neighbours of a
    sentence are from the same document — cross-agent edges get crowded out.
    This function finds the best cross-node sentence pairs and inserts edges
    that wouldn't otherwise exist, ensuring BFS can always traverse boundaries.

    Modifies adjacency in-place. O(N_nodes² × sentences_per_node).
    """
    node_ids = list(nodes.keys())
    if len(node_ids) < 2:
        return

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            src_node = nodes[node_ids[i]]
            dst_node = nodes[node_ids[j]]
            if not src_node.sentence_ids or not dst_node.sentence_ids:
                continue

            src_emb = embeddings[src_node.sentence_ids]  # (n, d)
            dst_emb = embeddings[dst_node.sentence_ids]  # (m, d)
            cross_sims = src_emb @ dst_emb.T             # (n, m)

            added = 0
            # Greedily pick best pairs until we've added k cross edges
            tmp = cross_sims.copy()
            while added < k:
                flat_idx = int(np.argmax(tmp))
                si, di = divmod(flat_idx, tmp.shape[1])
                sim = float(tmp[si, di])
                if sim <= 0:
                    break
                src_idx = src_node.sentence_ids[si]
                dst_idx = dst_node.sentence_ids[di]

                # Add edge in both directions if not already present
                existing_src = {idx for idx, _ in adjacency.get(src_idx, [])}
                if dst_idx not in existing_src:
                    adjacency.setdefault(src_idx, []).append((dst_idx, sim))
                    adjacency.setdefault(dst_idx, []).append((src_idx, sim))
                    added += 1

                tmp[si, di] = -1  # prevent re-selection
                if tmp.max() <= 0:
                    break


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
