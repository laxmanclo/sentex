"""SQLite persistence layer (optional).

Call graph.save(path) to persist; ContextGraph.load(path) to restore.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .graph import ContextGraph

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id              TEXT PRIMARY KEY,
    produced_by     TEXT NOT NULL,
    l0              TEXT,
    l2              TEXT,
    l3              TEXT,
    first_sentence  TEXT DEFAULT '',
    sentence_ids    TEXT NOT NULL,
    token_l0        INTEGER DEFAULT 0,
    token_l1        INTEGER DEFAULT 0,
    token_l2        INTEGER DEFAULT 0,
    token_l3        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sentences (
    idx         INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    idx         INTEGER PRIMARY KEY,
    vector      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS adjacency (
    src         INTEGER NOT NULL,
    dst         INTEGER NOT NULL,
    similarity  REAL NOT NULL,
    PRIMARY KEY (src, dst)
);

CREATE TABLE IF NOT EXISTS usage_boost (
    src         INTEGER NOT NULL,
    dst         INTEGER NOT NULL,
    boost       REAL NOT NULL,
    PRIMARY KEY (src, dst)
);
"""


def save(graph: "ContextGraph", path: str | Path) -> None:
    path = Path(path)
    con = sqlite3.connect(path)
    con.executescript(_SCHEMA)

    cur = con.cursor()

    # nodes
    for node in graph._nodes.values():
        cur.execute(
            """INSERT OR REPLACE INTO nodes
               (id, produced_by, l0, l2, l3, first_sentence, sentence_ids,
                token_l0, token_l1, token_l2, token_l3)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                node.id,
                node.produced_by,
                node.l0,
                node.l2,
                json.dumps(node.l3) if not isinstance(node.l3, str) else node.l3,
                node.first_sentence,
                json.dumps(node.sentence_ids),
                node.token_counts.l0,
                node.token_counts.l1,
                node.token_counts.l2,
                node.token_counts.l3,
            ),
        )

    # sentences + embeddings
    for meta in graph._metadata:
        i = meta.sentence_idx
        cur.execute(
            "INSERT OR REPLACE INTO sentences VALUES (?,?,?,?)",
            (i, graph._sentences[i], meta.node_id, meta.agent_id),
        )
        vec = graph._embeddings[i].tobytes()
        cur.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)", (i, vec))

    # adjacency
    for src, neighbors in graph._adjacency.items():
        for dst, sim in neighbors:
            cur.execute(
                "INSERT OR REPLACE INTO adjacency VALUES (?,?,?)", (src, dst, sim)
            )

    # usage boosts
    for (src, dst), boost in graph._usage_boost.items():
        cur.execute(
            "INSERT OR REPLACE INTO usage_boost VALUES (?,?,?)", (src, dst, boost)
        )

    con.commit()
    con.close()


def load(path: str | Path, graph: "ContextGraph") -> None:
    """Load persisted state into an existing (empty) ContextGraph."""
    from .types import ContextNode, L0L1L2L3TokenCounts, SentenceMetadata

    path = Path(path)
    if not path.exists():
        return

    con = sqlite3.connect(path)

    rows = con.execute("SELECT * FROM sentences ORDER BY idx").fetchall()
    if not rows:
        con.close()
        return

    n = len(rows)
    dim = graph.embedder.dim

    embeddings = np.zeros((n, dim), dtype=np.float32)
    sentences: list[str] = [""] * n
    metadata: list[SentenceMetadata] = [None] * n  # type: ignore[list-item]

    for idx, text, node_id, agent_id in rows:
        sentences[idx] = text
        metadata[idx] = SentenceMetadata(idx, node_id, agent_id)

    for idx, blob in con.execute("SELECT idx, vector FROM embeddings ORDER BY idx"):
        embeddings[idx] = np.frombuffer(blob, dtype=np.float32)

    adjacency: dict[int, list[tuple[int, float]]] = {}
    for src, dst, sim in con.execute("SELECT src, dst, similarity FROM adjacency"):
        adjacency.setdefault(src, []).append((dst, sim))

    usage_boost: dict[tuple[int, int], float] = {}
    for src, dst, boost in con.execute("SELECT src, dst, boost FROM usage_boost"):
        usage_boost[(src, dst)] = boost

    nodes: dict[str, ContextNode] = {}
    for row in con.execute(
        "SELECT id, produced_by, l0, l2, l3, first_sentence, sentence_ids,"
        " token_l0, token_l1, token_l2, token_l3 FROM nodes"
    ):
        nid, produced_by, l0, l2, l3_raw, first_sentence, sid_json, tl0, tl1, tl2, tl3 = row
        try:
            l3 = json.loads(l3_raw)
        except Exception:
            l3 = l3_raw
        nodes[nid] = ContextNode(
            id=nid,
            produced_by=produced_by,
            l3=l3,
            sentence_ids=json.loads(sid_json),
            l0=l0 or "",
            l2=l2 or "",
            first_sentence=first_sentence or "",
            token_counts=L0L1L2L3TokenCounts(l0=tl0, l1=tl1, l2=tl2, l3=tl3),
        )

    con.close()

    graph._embeddings = embeddings
    graph._sentences = sentences
    graph._metadata = metadata
    graph._adjacency = adjacency
    graph._usage_boost = usage_boost
    graph._nodes = nodes
