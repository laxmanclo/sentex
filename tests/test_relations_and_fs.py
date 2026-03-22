"""Tests for Relations and GraphFS."""
import numpy as np
import pytest

from sentex.core.relations import Relation, RelationIndex
from sentex.ingestion.embedder import Embedder
from sentex.core.graph import ContextGraph


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class _FakeEmbedder(Embedder):
    def __init__(self, dim=16):
        self._dim = dim
        self._model = object()
        self._cache = {}
        self.model_name = "fake"

    def embed(self, texts):
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(self._dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.stack(vecs)

    def embed_one(self, text):
        return self.embed([text])[0]

    @property
    def dim(self):
        return self._dim


def make_graph():
    g = ContextGraph(embedder=_FakeEmbedder())
    g.put("resources/search", "T-cells fight viruses. B-cells make antibodies.", agent_id="researcher")
    g.put("working/analysis", "Key finding: T-cells are central.", agent_id="analyst")
    g.put("working/script", "Here is the final script text.", agent_id="writer")
    g.put("memory/facts", "Macrophages engulf pathogens.", agent_id="memory")
    return g


# ------------------------------------------------------------------
# RelationIndex
# ------------------------------------------------------------------

def test_link_creates_relation():
    idx = RelationIndex()
    rel = idx.link("A", "B", kind="derived_from")
    assert rel.src == "A"
    assert rel.dst == "B"
    assert rel.kind == "derived_from"


def test_link_idempotent():
    idx = RelationIndex()
    idx.link("A", "B", kind="references")
    idx.link("A", "B", kind="references")
    assert len(idx) == 1


def test_link_different_kinds_different_relations():
    idx = RelationIndex()
    idx.link("A", "B", kind="references")
    idx.link("A", "B", kind="derived_from")
    assert len(idx) == 2


def test_unlink_removes():
    idx = RelationIndex()
    idx.link("A", "B", kind="references")
    removed = idx.unlink("A", "B")
    assert removed == 1
    assert len(idx) == 0


def test_unlink_by_kind():
    idx = RelationIndex()
    idx.link("A", "B", kind="references")
    idx.link("A", "B", kind="derived_from")
    idx.unlink("A", "B", kind="references")
    assert len(idx) == 1
    remaining = idx.neighbors("A", direction="out")
    assert remaining[0].kind == "derived_from"


def test_neighbors_out():
    idx = RelationIndex()
    idx.link("A", "B", kind="references")
    idx.link("A", "C", kind="summarizes")
    rels = idx.neighbors("A", direction="out")
    assert len(rels) == 2
    assert all(r.src == "A" for r in rels)


def test_neighbors_in():
    idx = RelationIndex()
    idx.link("A", "B", kind="references")
    rels = idx.neighbors("B", direction="in")
    assert len(rels) == 1
    assert rels[0].src == "A"


def test_neighbors_both():
    idx = RelationIndex()
    idx.link("A", "B")
    idx.link("C", "A")
    rels = idx.neighbors("A", direction="both")
    assert len(rels) == 2


def test_neighbor_ids():
    idx = RelationIndex()
    idx.link("A", "B")
    idx.link("A", "C")
    ids = idx.neighbor_ids("A", direction="out")
    assert set(ids) == {"B", "C"}


def test_to_list_from_list_roundtrip():
    idx = RelationIndex()
    idx.link("A", "B", kind="references", weight=0.9, created_by="agent1")
    idx.link("B", "C", kind="derived_from")
    data = idx.to_list()
    restored = RelationIndex.from_list(data)
    assert len(restored) == 2
    rels_a = restored.neighbors("A", direction="out")
    assert rels_a[0].weight == 0.9
    assert rels_a[0].created_by == "agent1"


# ------------------------------------------------------------------
# Graph-level link/unlink
# ------------------------------------------------------------------

def test_graph_link():
    g = make_graph()
    rel = g.link("working/analysis", "resources/search", kind="derived_from")
    assert rel.src == "working/analysis"
    assert rel.dst == "resources/search"
    assert rel.kind == "derived_from"


def test_graph_unlink():
    g = make_graph()
    g.link("working/analysis", "resources/search")
    removed = g.unlink("working/analysis", "resources/search")
    assert removed == 1


def test_graph_neighbors():
    g = make_graph()
    g.link("working/script", "working/analysis", kind="derived_from")
    g.link("working/script", "resources/search", kind="references")
    rels = g.neighbors("working/script", direction="out")
    assert len(rels) == 2


def test_stats_includes_relations():
    g = make_graph()
    g.link("working/analysis", "resources/search", kind="derived_from")
    s = g.stats()
    assert s["relations"] == 1


# ------------------------------------------------------------------
# GraphFS
# ------------------------------------------------------------------

def test_fs_property_returns_graphfs():
    from sentex.core.fs import GraphFS
    g = make_graph()
    assert isinstance(g.fs, GraphFS)


def test_fs_ls_root():
    g = make_graph()
    entries = g.fs.ls("/")
    names = {e.name for e in entries}
    assert "resources" in names
    assert "working" in names
    assert "memory" in names


def test_fs_ls_scope():
    g = make_graph()
    entries = g.fs.ls("working")
    paths = {e.path for e in entries}
    assert "working/analysis" in paths
    assert "working/script" in paths
    assert "resources/search" not in paths


def test_fs_ls_returns_dirs_and_files():
    g = ContextGraph(embedder=_FakeEmbedder())
    g.put("resources/docs/paper1", "Neural networks learn representations.", agent_id="x")
    g.put("resources/docs/paper2", "Transformers use attention mechanisms.", agent_id="x")
    g.put("resources/search", "Search result text.", agent_id="x")

    entries = g.fs.ls("resources")
    kinds = {e.kind for e in entries}
    assert "dir" in kinds   # resources/docs/ should appear as dir
    assert "file" in kinds  # resources/search should appear as file


def test_fs_stat_existing_node():
    g = make_graph()
    entry = g.fs.stat("resources/search")
    assert entry is not None
    assert entry.path == "resources/search"
    assert entry.kind == "file"
    assert entry.sentences > 0
    assert entry.produced_by == "researcher"


def test_fs_stat_missing_node():
    g = make_graph()
    assert g.fs.stat("nonexistent/node") is None


def test_fs_find_glob():
    g = make_graph()
    results = g.fs.find("working/*")
    assert "working/analysis" in results
    assert "working/script" in results
    assert "resources/search" not in results


def test_fs_find_pattern():
    g = make_graph()
    results = g.fs.find("*search*")
    assert "resources/search" in results


def test_fs_find_scope():
    g = make_graph()
    results = g.fs.find("*", scope="working")
    assert all(r.startswith("working/") for r in results)


def test_fs_tree_structure():
    g = make_graph()
    tree = g.fs.tree()
    assert tree["kind"] == "dir"
    assert "children" in tree
    assert len(tree["children"]) > 0


def test_fs_link_via_fs():
    g = make_graph()
    rel = g.fs.link("working/analysis", "resources/search", kind="references")
    assert rel.kind == "references"
    assert len(g._relations) == 1


def test_fs_repr():
    g = make_graph()
    r = repr(g.fs)
    assert "GraphFS" in r


# ------------------------------------------------------------------
# Convergence in retrieval
# ------------------------------------------------------------------

def test_retrieve_convergence_flag():
    """retrieve_l1 should return a converged bool."""
    from sentex.retrieval.engine import retrieve_l1
    import numpy as np

    dim = 16
    n = 20
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms

    query = embeddings[0]
    sentences = [f"sentence {i}" for i in range(n)]
    adjacency = {i: [(i + 1, 0.9)] for i in range(n - 1)}

    result = retrieve_l1(query, embeddings, sentences, adjacency, budget_tokens=5000)
    assert len(result) == 3   # (sentences, confidence, converged)
    sents, conf, converged = result
    assert isinstance(sents, list)
    assert isinstance(conf, float)
    assert isinstance(converged, bool)
