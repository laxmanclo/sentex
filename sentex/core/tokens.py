"""Token counting utilities.

The encoder is selected by the SENTEX_TOKEN_ENCODING env var (or the legacy
ENGRAM_TOKEN_MODEL var). Supported values:

  cl100k_base   — GPT-3.5/GPT-4 / compatible with most open models (default)
  o200k_base    — GPT-4o, newer OpenAI models
  gpt2          — rough estimate for models without a tiktoken encoding

Set at process start; call reset_encoder() to switch mid-process (e.g. in tests).
"""
from __future__ import annotations

import os

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken
        # SENTEX_TOKEN_ENCODING accepts a tiktoken encoding name directly.
        # Fall back to ENGRAM_TOKEN_MODEL for legacy compat, then cl100k_base.
        enc_name = os.getenv("SENTEX_TOKEN_ENCODING")
        if enc_name:
            try:
                _encoder = tiktoken.get_encoding(enc_name)
                return _encoder
            except Exception:
                pass
        model = os.getenv("ENGRAM_TOKEN_MODEL")
        if model:
            try:
                _encoder = tiktoken.encoding_for_model(model)
                return _encoder
            except KeyError:
                pass
        # Default: cl100k_base — accurate for GPT-4/Claude/most LLMs (~same tokenizer)
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def reset_encoder() -> None:
    """Clear the cached encoder so the next call re-reads env vars."""
    global _encoder
    _encoder = None


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def count_tokens_list(texts: list[str]) -> int:
    return sum(count_tokens(t) for t in texts)
