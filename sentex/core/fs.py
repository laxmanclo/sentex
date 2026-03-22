"""GraphFS — filesystem-style API over ContextGraph.

Exposes the node namespace (resources/, memory/, working/) as an intuitive
virtual filesystem. Useful for building UIs, debugging tools, and pipeline
introspection.

Usage:
    from sentex import ContextGraph

    graph = ContextGraph()
    graph.put("resources/research", "...")
    graph.put("working/analysis", "...")

    fs = graph.fs

    fs.ls()                        # list all scopes
    fs.ls("resources")             # list nodes under resources/
    fs.tree()                      # hierarchical dict of all nodes
    fs.stat("resources/research")  # metadata for a single node
    fs.find("resources/*search*")  # glob-style search over node IDs
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ContextGraph
    from .relations import Relation


@dataclass
class FSEntry:
    name: str               # the last path component
    path: str               # full node ID (or scope name for dirs)
    kind: str               # "dir" | "file"
    sentences: int = 0
    token_l0: int = 0
    token_l1: int = 0
    token_l2: int = 0
    token_l3: int = 0
    produced_by: str = ""
    l0_summary: str = ""    # first sentence / L0 identity

    def __repr__(self) -> str:
        if self.kind == "dir":
            return f"<dir  {self.path}/>"
        return (
            f"<file {self.path} | {self.sentences} sentences | "
            f"L1={self.token_l1}tok | by={self.produced_by}>"
        )


class GraphFS:
    """Virtual filesystem view over a ContextGraph."""

    def __init__(self, graph: "ContextGraph") -> None:
        self._g = graph

    # ------------------------------------------------------------------
    # Directory listing
    # ------------------------------------------------------------------

    def ls(self, path: str = "/") -> list[FSEntry]:
        """List entries at path.

        ls("/")            → top-level scopes (resources, working, memory, ...)
        ls("resources")    → all nodes under resources/
        ls("resources/sub")→ all nodes under resources/sub/
        """
        path = path.strip("/")
        entries: list[FSEntry] = []

        if not path or path == "":
            # Root: return unique top-level scopes (everything before first /)
            scopes: set[str] = set()
            for node_id in self._g._nodes:
                top = node_id.split("/")[0]
                scopes.add(top)
            for scope in sorted(scopes):
                entries.append(FSEntry(name=scope, path=scope, kind="dir"))
            return entries

        # Scope or deeper path
        prefix = path + "/"
        seen_subdirs: set[str] = set()

        for node_id, node in self._g._nodes.items():
            if not node_id.startswith(prefix) and node_id != path:
                continue
            remainder = node_id[len(prefix):]
            if "/" in remainder:
                # There's a deeper sub-dir; emit the sub-dir, not the file
                subdir = remainder.split("/")[0]
                key = prefix + subdir
                if key not in seen_subdirs:
                    seen_subdirs.add(key)
                    entries.append(FSEntry(name=subdir, path=key, kind="dir"))
            else:
                # Direct child
                entries.append(self._node_to_entry(node_id, node))

        return sorted(entries, key=lambda e: (e.kind, e.name))

    def tree(self, path: str = "/", depth: int = 3) -> dict:
        """Return a nested dict representing the directory tree.

        {"name": "resources", "kind": "dir", "children": [
            {"name": "research", "path": "resources/research", "kind": "file",
             "sentences": 12, "token_l1": 430},
            ...
        ]}
        """
        path = path.strip("/")
        return self._build_tree(path, depth, 0)

    def _build_tree(self, path: str, max_depth: int, current_depth: int) -> dict:
        entries = self.ls(path or "/")
        name = path.split("/")[-1] if "/" in path else (path or "/")
        node = {
            "name": name,
            "path": path or "/",
            "kind": "dir",
            "children": [],
        }
        for e in entries:
            if e.kind == "dir" and current_depth < max_depth - 1:
                node["children"].append(
                    self._build_tree(e.path, max_depth, current_depth + 1)
                )
            else:
                child: dict = {
                    "name": e.name,
                    "path": e.path,
                    "kind": e.kind,
                }
                if e.kind == "file":
                    child.update({
                        "sentences": e.sentences,
                        "token_l1": e.token_l1,
                        "produced_by": e.produced_by,
                        "l0_summary": e.l0_summary[:80] if e.l0_summary else "",
                    })
                node["children"].append(child)
        return node

    # ------------------------------------------------------------------
    # Node metadata
    # ------------------------------------------------------------------

    def stat(self, path: str) -> FSEntry | None:
        """Return metadata for a single node. None if not found."""
        path = path.strip("/")
        node = self._g._nodes.get(path)
        if node is None:
            return None
        return self._node_to_entry(path, node)

    def _node_to_entry(self, node_id: str, node) -> FSEntry:
        name = node_id.split("/")[-1]
        return FSEntry(
            name=name,
            path=node_id,
            kind="file",
            sentences=len(node.sentence_ids),
            token_l0=node.token_counts.l0,
            token_l1=node.token_counts.l1,
            token_l2=node.token_counts.l2,
            token_l3=node.token_counts.l3,
            produced_by=node.produced_by,
            l0_summary=node.l0 or node.first_sentence,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find(self, pattern: str, scope: str | None = None) -> list[str]:
        """Glob-style search over node IDs.

        find("resources/*search*")  → all nodes under resources/ matching *search*
        find("*analysis*")          → any node whose ID contains 'analysis'
        """
        results = []
        for node_id in self._g._nodes:
            if scope and not node_id.startswith(scope.rstrip("/") + "/"):
                continue
            if fnmatch.fnmatch(node_id, pattern):
                results.append(node_id)
        return sorted(results)

    # ------------------------------------------------------------------
    # Relations (delegates to graph._relations if present)
    # ------------------------------------------------------------------

    def link(
        self,
        src: str,
        dst: str,
        kind: str = "references",
        weight: float = 1.0,
        created_by: str = "__fs__",
    ) -> "Relation":
        return self._g.link(src, dst, kind=kind, weight=weight, created_by=created_by)

    def unlink(self, src: str, dst: str, kind: str | None = None) -> int:
        return self._g.unlink(src, dst, kind=kind)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nodes = self._g.node_count
        scopes = len({n.split("/")[0] for n in self._g._nodes})
        return f"<GraphFS {nodes} nodes across {scopes} scopes>"
