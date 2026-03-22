"""Bidirectional node relations for ContextGraph.

Relations are explicit semantic links between nodes, complementing the
implicit KNN sentence-level edges. They express provenance, dependency,
and conceptual relationships:

    graph.link("working/analysis", "resources/search", kind="derived_from")
    graph.link("working/script",   "working/analysis", kind="derived_from")
    graph.link("resources/papers", "resources/search", kind="references")

Graph traversal can optionally follow relation edges to expand retrieval scope.

Relation kinds (convention — not enforced):
    "derived_from"  — this node's content was produced using the target
    "references"    — explicit citation or link
    "summarizes"    — this node is a summary of the target
    "contradicts"   — conflicting information
    "extends"       — builds on the target's content
    "related"       — generic semantic relation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

RelationKind = Literal[
    "derived_from", "references", "summarizes",
    "contradicts", "extends", "related",
] | str


@dataclass
class Relation:
    src: str            # source node ID
    dst: str            # destination node ID
    kind: RelationKind
    weight: float = 1.0
    created_by: str = "__system__"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "weight": self.weight,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(
            src=d["src"],
            dst=d["dst"],
            kind=d["kind"],
            weight=d.get("weight", 1.0),
            created_by=d.get("created_by", "__system__"),
            created_at=d.get("created_at", time.time()),
        )


class RelationIndex:
    """In-memory bidirectional relation graph.

    Relations are keyed by (src, dst, kind) — the same pair can have
    multiple relations of different kinds.
    """

    def __init__(self) -> None:
        # forward: src → list[Relation]
        self._forward: dict[str, list[Relation]] = {}
        # backward: dst → list[Relation]
        self._backward: dict[str, list[Relation]] = {}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def link(
        self,
        src: str,
        dst: str,
        kind: RelationKind = "related",
        weight: float = 1.0,
        created_by: str = "__system__",
    ) -> Relation:
        """Create a relation from src → dst. Idempotent for same (src, dst, kind)."""
        # Check for existing
        for r in self._forward.get(src, []):
            if r.dst == dst and r.kind == kind:
                r.weight = weight
                return r

        rel = Relation(src=src, dst=dst, kind=kind, weight=weight, created_by=created_by)
        self._forward.setdefault(src, []).append(rel)
        self._backward.setdefault(dst, []).append(rel)
        return rel

    def unlink(
        self,
        src: str,
        dst: str,
        kind: RelationKind | None = None,
    ) -> int:
        """Remove relations matching (src, dst) and optionally kind. Returns count removed."""
        removed = 0
        if src in self._forward:
            before = self._forward[src]
            if kind is None:
                kept = [r for r in before if r.dst != dst]
            else:
                kept = [r for r in before if not (r.dst == dst and r.kind == kind)]
            self._forward[src] = kept
            removed += len(before) - len(kept)

        if dst in self._backward:
            before = self._backward[dst]
            if kind is None:
                kept = [r for r in before if r.src != src]
            else:
                kept = [r for r in before if not (r.src == src and r.kind == kind)]
            self._backward[dst] = kept
        return removed

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        direction: Literal["out", "in", "both"] = "both",
        kind: RelationKind | None = None,
    ) -> list[Relation]:
        """Return all relations for a node. Filter by kind if provided."""
        rels: list[Relation] = []
        if direction in ("out", "both"):
            rels.extend(self._forward.get(node_id, []))
        if direction in ("in", "both"):
            rels.extend(self._backward.get(node_id, []))

        if kind is not None:
            rels = [r for r in rels if r.kind == kind]
        return rels

    def neighbor_ids(
        self,
        node_id: str,
        direction: Literal["out", "in", "both"] = "both",
        kind: RelationKind | None = None,
    ) -> list[str]:
        """Return just the node IDs of neighbors."""
        ids: list[str] = []
        for r in self.neighbors(node_id, direction, kind):
            ids.append(r.dst if r.src == node_id else r.src)
        return ids

    def all_relations(self) -> list[Relation]:
        """Return all relations (de-duplicated — forward direction only)."""
        seen: set[tuple[str, str, str]] = set()
        result: list[Relation] = []
        for rels in self._forward.values():
            for r in rels:
                key = (r.src, r.dst, r.kind)
                if key not in seen:
                    seen.add(key)
                    result.append(r)
        return result

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.all_relations()]

    @classmethod
    def from_list(cls, data: list[dict]) -> "RelationIndex":
        idx = cls()
        for d in data:
            rel = Relation.from_dict(d)
            idx._forward.setdefault(rel.src, []).append(rel)
            idx._backward.setdefault(rel.dst, []).append(rel)
        return idx

    def __len__(self) -> int:
        return sum(len(v) for v in self._forward.values())
