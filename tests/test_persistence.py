import tempfile
from pathlib import Path

import numpy as np

from sentex.graph import ContextGraph
from sentex.embedder import Embedder
from sentex import save, load


class _FakeEmbedder(Embedder):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._model = object()
        self._cache: dict = {}
        self.model_name = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(self._dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.stack(vecs)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._dim


def test_save_and_load_roundtrip():
    g = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
    g.ingest("alpha", "First sentence here. Second sentence here.", "a1", generate_summaries=False)
    g.ingest("beta", "Third sentence here.", "a2", generate_summaries=False)

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        save(g, db)

        g2 = ContextGraph(embedder=_FakeEmbedder(), knn_k=3)
        load(db, g2)

        assert g2.node_count == 2
        assert g2.sentence_count == g.sentence_count
        assert "alpha" in g2._nodes
        assert "beta" in g2._nodes
        assert len(g2._adjacency) == g.sentence_count
