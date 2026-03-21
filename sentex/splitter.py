"""Sentence splitter backed by NLTK's Punkt tokenizer.

Special handling (pre/post NLTK):
- Fenced code blocks  → kept as single atomic units
- Markdown list items → each item is atomic
- Everything else     → NLTK sent_tokenize

First import downloads the Punkt model if not already present.
"""
from __future__ import annotations

import re


_FENCED_CODE = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)", re.MULTILINE)
_LIST_ITEM   = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+.+$", re.MULTILINE)
_PLACEHOLDER = "\x00BLOCK{}\x00"


def _ensure_nltk() -> None:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences / atomic units."""
    if not text or not text.strip():
        return []

    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    placeholders: dict[str, str] = {}

    # 1. Stash fenced code blocks
    def _stash_code(m: re.Match) -> str:
        key = _PLACEHOLDER.format(len(placeholders))
        placeholders[key] = m.group(0).strip()
        return f" {key} "

    working = _FENCED_CODE.sub(_stash_code, text)

    # 2. Walk line by line; list items are atomic, prose is buffered
    segments: list[str] = []
    prose_buf: list[str] = []

    def _flush() -> None:
        prose = " ".join(prose_buf).strip()
        prose_buf.clear()
        if prose:
            segments.extend(sent_tokenize(prose))

    for line in working.splitlines():
        stripped = line.strip()
        if not stripped:
            _flush()
            continue
        if _LIST_ITEM.match(line):
            _flush()
            segments.append(stripped)
        else:
            prose_buf.append(stripped)

    _flush()

    # 3. Restore placeholders
    result: list[str] = []
    for seg in segments:
        for key, val in placeholders.items():
            seg = seg.replace(key, val)
        seg = seg.strip()
        if seg:
            result.append(seg)

    return result
