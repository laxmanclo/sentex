"""Sentence embedding wrapper.

Default: all-MiniLM-L6-v2 (384 dims, local, ~80 MB).
Swappable via ENGRAM_EMBEDDING_MODEL env var or by passing model= to Embedder().
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np


_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or os.getenv("ENGRAM_EMBEDDING_MODEL", _DEFAULT_MODEL)
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, D) float32 array. Cached by content hash."""
        self._load()
        results: list[np.ndarray] = []
        to_compute: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            key = hashlib.md5(t.encode()).hexdigest()
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                to_compute.append((i, t, key))

        if to_compute:
            vecs = self._model.encode(
                [t for _, t, _ in to_compute],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for (i, t, key), vec in zip(to_compute, vecs):
                self._cache[key] = vec
                results.append((i, vec))

        results.sort(key=lambda x: x[0])
        return np.stack([v for _, v in results], axis=0).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
