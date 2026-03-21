"""L0 and L2 generation via LiteLLM.

Non-blocking: callers await generate_l0 / generate_l2.
Any LiteLLM-compatible model works; defaults to gpt-4o-mini.
"""
from __future__ import annotations

import asyncio
import os


_DEFAULT_MODEL = os.getenv("ENGRAM_LLM_MODEL", "gpt-4o-mini")


async def generate_l0(content: str, model: str | None = None) -> str:
    """Generate a ~50-token identity sentence for *content*."""
    import litellm

    prompt = (
        "In one sentence (under 50 words), describe what the following content is — "
        "not what it says, but what kind of thing it is and what topic it covers.\n\n"
        f"Content:\n{content[:4000]}"
    )
    resp = await asyncio.to_thread(
        litellm.completion,
        model=model or _DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


async def generate_l2(content: str, model: str | None = None) -> str:
    """Generate a ~2000-token coherent narrative summary of *content*."""
    import litellm

    prompt = (
        "Write a clear, coherent summary of the following content. "
        "Include the key points, findings, and structure. "
        "Aim for around 300-400 words.\n\n"
        f"Content:\n{content[:12000]}"
    )
    resp = await asyncio.to_thread(
        litellm.completion,
        model=model or _DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()
