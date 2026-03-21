"""Token counting utilities."""
from __future__ import annotations

import os


_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken
        model = os.getenv("ENGRAM_TOKEN_MODEL", "gpt-4o")
        try:
            _encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def count_tokens_list(texts: list[str]) -> int:
    return sum(count_tokens(t) for t in texts)
