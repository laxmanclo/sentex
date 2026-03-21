"""Tests for hotness scoring."""
import math
import time

from sentex.scoring import HotnessScore, compute_hotness


def test_cold_edge_is_zero():
    score = HotnessScore(hit_count=0)
    assert compute_hotness(score) == 0.0


def test_single_hit_nonzero():
    score = HotnessScore(hit_count=1, last_hit_at=time.time())
    assert compute_hotness(score) > 0.0


def test_more_hits_higher_hotness():
    now = time.time()
    s1 = HotnessScore(hit_count=1, last_hit_at=now)
    s10 = HotnessScore(hit_count=10, last_hit_at=now)
    s50 = HotnessScore(hit_count=50, last_hit_at=now)
    assert compute_hotness(s1) < compute_hotness(s10) < compute_hotness(s50)


def test_hotness_decays_with_age():
    now = time.time()
    fresh = HotnessScore(hit_count=5, last_hit_at=now)
    stale = HotnessScore(hit_count=5, last_hit_at=now - 86_400 * 7)  # 7 days ago
    assert compute_hotness(fresh) > compute_hotness(stale)


def test_hotness_bounded():
    score = HotnessScore(hit_count=1000, last_hit_at=time.time())
    h = compute_hotness(score)
    assert 0.0 <= h <= 1.0


def test_hit_updates_count_and_time():
    score = HotnessScore()
    t_before = time.time()
    score.hit()
    t_after = time.time()
    assert score.hit_count == 1
    assert t_before <= score.last_hit_at <= t_after


def test_to_from_dict_roundtrip():
    score = HotnessScore(hit_count=7, last_hit_at=12345.6)
    d = score.to_dict()
    restored = HotnessScore.from_dict(d)
    assert restored.hit_count == 7
    assert restored.last_hit_at == 12345.6


def test_graph_uses_hotness_boost(tmp_path):
    """Graph._usage_boost should be HotnessScore dicts after mark_used."""
    import numpy as np
    from sentex.graph import ContextGraph
    from sentex.embedder import Embedder

    class _FakeEmbedder(Embedder):
        def __init__(self):
            self._dim = 16
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

    graph = ContextGraph(embedder=_FakeEmbedder())
    graph.put("resources/a", "Sentence one about AI. Sentence two about agents.", agent_id="x")
    graph.put("working/b", "Result about machine learning models.", agent_id="y")
    graph.used("resources/a")

    assert len(graph._usage_boost) > 0
    for key, val in graph._usage_boost.items():
        assert isinstance(val, HotnessScore)
        assert val.hit_count >= 1
