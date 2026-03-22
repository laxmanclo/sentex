"""MemoryStore — cross-run persistent memory.

Survives between pipeline runs. On each Session.commit():
  - Edge weights (which sentences were co-useful) are written to SQLite
  - Node summaries (L0/L2) are cached so they never need to be regenerated
  - Usage counts per node are tracked

On the next run, the graph is pre-seeded with these weights, so retrieval
improves over time without any explicit feedback from the developer.

Scopes mirror OpenViking's namespace model:
  resources/   → shared knowledge (docs, search results, reference data)
  memory/      → extracted learnings from past runs
  working/     → ephemeral within a pipeline run (not persisted across runs)
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    scope       TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    l0          TEXT DEFAULT '',
    l2          TEXT DEFAULT '',
    l3          TEXT,
    produced_by TEXT NOT NULL,
    created_at  REAL NOT NULL,
    run_count   INTEGER DEFAULT 0,
    PRIMARY KEY (scope, node_id)
);

CREATE TABLE IF NOT EXISTS sentences (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    agent_id    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    sentence_db_id  INTEGER PRIMARY KEY,
    vector          BLOB NOT NULL,
    FOREIGN KEY (sentence_db_id) REFERENCES sentences(id)
);

CREATE TABLE IF NOT EXISTS edge_weights (
    src_id      INTEGER NOT NULL,
    dst_id      INTEGER NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.0,
    hit_count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (src_id, dst_id),
    FOREIGN KEY (src_id) REFERENCES sentences(id),
    FOREIGN KEY (dst_id) REFERENCES sentences(id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    started_at  REAL NOT NULL,
    committed_at REAL,
    agent_ids   TEXT NOT NULL,
    node_ids    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source_run  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    UNIQUE (scope, key)
);

CREATE INDEX IF NOT EXISTS idx_sentences_node ON sentences(scope, node_id);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edge_weights(src_id);
"""


class MemoryStore:
    """
    SQLite-backed cross-run memory.

    Usage:
        store = MemoryStore("./sentex.db")
        pipeline = Pipeline(store=store)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.executescript(_SCHEMA)
        self._con.commit()

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def record_session(
        self,
        session_id: str,
        query: str,
        agent_ids: list[str],
        node_ids: list[str],
    ) -> None:
        self._con.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, query, started_at, agent_ids, node_ids)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, query, time.time(),
             json.dumps(agent_ids), json.dumps(node_ids)),
        )
        self._con.commit()

    def commit_session(self, session_id: str) -> None:
        self._con.execute(
            "UPDATE sessions SET committed_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Node cache (L0/L2 summaries — expensive to regenerate)
    # ------------------------------------------------------------------

    def save_node_summary(
        self,
        scope: str,
        node_id: str,
        l0: str,
        l2: str,
        l3: Any,
        produced_by: str,
    ) -> None:
        l3_str = l3 if isinstance(l3, str) else json.dumps(l3)
        self._con.execute(
            """INSERT INTO nodes (scope, node_id, l0, l2, l3, produced_by, created_at, run_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT (scope, node_id) DO UPDATE SET
                 l0 = CASE WHEN excluded.l0 != '' THEN excluded.l0 ELSE l0 END,
                 l2 = CASE WHEN excluded.l2 != '' THEN excluded.l2 ELSE l2 END,
                 l3 = excluded.l3,
                 run_count = run_count + 1""",
            (scope, node_id, l0, l2, l3_str, produced_by, time.time()),
        )
        self._con.commit()

    def load_node_summary(self, scope: str, node_id: str) -> dict | None:
        row = self._con.execute(
            "SELECT l0, l2, l3 FROM nodes WHERE scope = ? AND node_id = ?",
            (scope, node_id),
        ).fetchone()
        if not row:
            return None
        l0, l2, l3_str = row
        try:
            l3 = json.loads(l3_str)
        except Exception:
            l3 = l3_str
        return {"l0": l0, "l2": l2, "l3": l3}

    # ------------------------------------------------------------------
    # Edge weight persistence
    # ------------------------------------------------------------------

    def boost_edge(self, src_db_id: int, dst_db_id: int, amount: float = 0.05) -> None:
        self._con.execute(
            """INSERT INTO edge_weights (src_id, dst_id, weight, hit_count)
               VALUES (?, ?, ?, 1)
               ON CONFLICT (src_id, dst_id) DO UPDATE SET
                 weight = weight + excluded.weight,
                 hit_count = hit_count + 1""",
            (src_db_id, dst_db_id, amount),
        )

    def flush_edge_boosts(self) -> None:
        self._con.commit()

    def load_edge_boosts(self) -> dict[tuple[int, int], float]:
        rows = self._con.execute(
            "SELECT src_id, dst_id, weight FROM edge_weights"
        ).fetchall()
        return {(r[0], r[1]): r[2] for r in rows}

    # ------------------------------------------------------------------
    # Memory entries (extracted learnings — key/value per scope)
    # ------------------------------------------------------------------

    def write_memory(self, scope: str, key: str, value: str, source_run: str) -> None:
        self._con.execute(
            """INSERT INTO memory_entries (scope, key, value, source_run, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (scope, key) DO UPDATE SET
                 value = excluded.value,
                 source_run = excluded.source_run,
                 created_at = excluded.created_at""",
            (scope, key, value, source_run, time.time()),
        )
        self._con.commit()

    def read_memory(self, scope: str) -> dict[str, str]:
        rows = self._con.execute(
            "SELECT key, value FROM memory_entries WHERE scope = ?", (scope,)
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def all_sessions(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT session_id, query, started_at, committed_at FROM sessions ORDER BY started_at DESC"
        ).fetchall()
        return [
            {"session_id": r[0], "query": r[1], "started_at": r[2], "committed_at": r[3]}
            for r in rows
        ]

    def close(self) -> None:
        self._con.close()
