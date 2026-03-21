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
    ContextNode,
    L0L1L2L3TokenCounts,
    Read,
    SentenceMetadata,
)
from .retrieval import retrieve_l1


_LAYER_ORDER = ["l3", "l2", "l1", "l0"]


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

        # Global sentence store
        self._embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
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
        generate_summaries: bool = True,
    ) -> ContextNode:
        """Ingest *content* as a new ContextNode.

        Splits into sentences, embeds, updates KNN.
        Schedules async L0/L2 generation if generate_summaries=True.
        """
        text = content if isinstance(content, str) else str(content)
        sentences = split_sentences(text)

        start_idx = len(self._sentences)

        # Embed new sentences
        if sentences:
            vecs = self.embedder.embed(sentences)
            if self._embeddings.shape[0] == 0:
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
            token_counts=L0L1L2L3TokenCounts(l1=l1_tokens, l3=l3_tokens),
        )
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

        Handles layer fallback automatically.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return "", layer, 0.0

        if layer == "l3":
            txt = node.l3 if isinstance(node.l3, str) else str(node.l3)
            return txt, "l3", 1.0

        if layer == "l2":
            return node.l2 or node.l0, "l2", 1.0

        if layer == "l0":
            return node.l0, "l0", 1.0

        # layer == "l1"
        if not node.sentence_ids or len(self._embeddings) == 0:
            return node.l2 or node.l0, fallback, 0.0

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
            return node.l2 or node.l0, fallback, confidence

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
                # Fall back one layer at a time
                fallback_order = _LAYER_ORDER[_LAYER_ORDER.index(layer_used) + 1 :]
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
                    # Truncate l0 to fit
                    node = self._nodes[node_id]
                    content = node.l0[:remaining * 4]  # rough char estimate
                    layer_used = "l0"
                    content_tokens = min(count_tokens(content), remaining)
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
        """Boost edge weights for sentences in nodes that were marked useful."""
        for node_id in used_ids:
            node = self._nodes.get(node_id)
            if node is None:
                continue
            ids = set(node.sentence_ids)
            for i in node.sentence_ids:
                for j, _ in self._adjacency.get(i, []):
                    if j in ids:
                        self._usage_boost[(i, j)] = (
                            self._usage_boost.get((i, j), 0.0) + boost
                        )

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
