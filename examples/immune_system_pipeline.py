"""
End-to-end example: 3-agent pipeline about immune systems.

Agents:
  1. researcher  — "searches" for content (mocked), writes search-results
  2. analyst     — reads search-results at L1 (sentence-level), writes analysis
  3. writer      — reads search-results (L1) + analysis (L2), writes final-script

Run:
    OPENAI_API_KEY=sk-... python examples/immune_system_pipeline.py

Or with a mock LLM (no API key needed):
    python examples/immune_system_pipeline.py --mock
"""

import asyncio
import sys

from sentex import Pipeline, Read

# ------------------------------------------------------------------
# Mock LLM (used when --mock flag is passed)
# ------------------------------------------------------------------

async def mock_llm(prompt: str) -> str:
    """Fake LLM that echoes back a canned response."""
    if "bullet" in prompt.lower() or "analyst" in prompt.lower():
        return (
            "Key findings:\n"
            "- T-cells and B-cells are the primary adaptive immune responders.\n"
            "- Macrophages act as first responders and antigen presenters.\n"
            "- The adaptive immune response creates immunological memory.\n"
            "- Fever is a systemic response that inhibits pathogen replication."
        )
    return (
        "The immune system is a remarkable defense network. "
        "T-cells hunt and destroy infected cells while B-cells manufacture "
        "targeted antibodies. Macrophages patrol tissues, engulfing debris "
        "and signaling danger. Together they form a layered shield — "
        "immediate innate defenses followed by a precision adaptive strike "
        "that leaves lasting memory against future invaders."
    )


# ------------------------------------------------------------------
# Fake search data (replaces a real web search)
# ------------------------------------------------------------------

SEARCH_RESULTS = """
The immune system is composed of two main branches: innate and adaptive immunity.
Innate immunity provides immediate, non-specific defense against pathogens.
The adaptive immune system mounts targeted responses to specific antigens.

T-cells are lymphocytes that develop in the thymus and play a central role in adaptive immunity.
Cytotoxic T-cells (CD8+) directly kill infected or cancerous cells.
Helper T-cells (CD4+) coordinate the immune response by activating other immune cells.

B-cells produce antibodies — proteins that bind to specific antigens on pathogens.
When a B-cell encounters its matching antigen, it differentiates into plasma cells.
Plasma cells secrete large quantities of antibodies into the bloodstream.
Memory B-cells persist long after an infection and enable faster future responses.

Macrophages are phagocytic cells that engulf and digest cellular debris and pathogens.
They also present antigen fragments on MHC-II molecules to activate helper T-cells.
Natural killer (NK) cells destroy virus-infected cells without prior sensitisation.

Fever is triggered by pyrogens released by macrophages during infection.
Elevated body temperature inhibits bacterial replication and accelerates immune activity.
The complement system consists of proteins that mark pathogens for destruction.

Vaccines work by training the immune system to recognise a pathogen without causing disease.
Immunological memory means a second exposure triggers a faster, stronger response.
Autoimmune disorders occur when the immune system mistakenly attacks the body's own tissue.
"""


# ------------------------------------------------------------------
# Pipeline definition
# ------------------------------------------------------------------

pipeline = Pipeline()


@pipeline.agent(
    id="researcher",
    writes=["search-results"],
    token_budget=8000,
)
async def researcher(ctx):
    # In production: call a real search API here
    # The return value is automatically ingested as "search-results"
    print(f"  [researcher] ingesting {len(SEARCH_RESULTS.split(chr(10)))} lines of search data")
    return SEARCH_RESULTS


@pipeline.agent(
    id="analyst",
    reads=[
        Read("search-results", layer="l1", budget=2000),  # sentence-graph retrieval
    ],
    writes=["analysis"],
    token_budget=3000,
)
async def analyst(ctx):
    sentences = ctx["search-results"]
    print(f"  [analyst] received {len(sentences)} sentences from graph (L1)")
    print(f"  [analyst] token utilisation: {ctx.token_utilization:.0%}")

    prompt = (
        "You are a science analyst. Given these retrieved facts about the immune system, "
        "extract the 4 most important insights as bullet points.\n\n"
        + ctx.render()
    )
    return await ctx.llm(prompt)


@pipeline.agent(
    id="writer",
    reads=[
        Read("search-results", layer="l1", budget=1500),
        Read("analysis",       layer="l2", budget=800),   # summary of analysis
    ],
    writes=["final-script"],
    token_budget=4000,
)
async def writer(ctx):
    print(f"  [writer] layers used: {ctx.assembled.layers_used}")
    print(f"  [writer] confidence: {ctx.assembled.confidence}")

    prompt = (
        "Write a compelling 60-second video script about the immune system "
        "for a general audience. Use the provided research and analysis.\n\n"
        + ctx.render()
    )
    return await ctx.llm(prompt)


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

async def main() -> None:
    use_mock = "--mock" in sys.argv
    llm = mock_llm if use_mock else "gpt-4o"

    print(f"\nRunning immune system pipeline ({'mock LLM' if use_mock else 'gpt-4o'})")
    print("=" * 60)

    result = await pipeline.run(
        query="explain how the immune system works for a general audience",
        llm=llm,
        generate_summaries=False,  # set True to generate L0/L2 via LLM
    )

    print("\n" + result.summary())

    print("\n--- Final Script ---")
    print(result.outputs.get("final-script", "(no output)"))

    print("\n--- Graph state ---")
    print(f"  nodes:     {result.graph.node_count}")
    print(f"  sentences: {result.graph.sentence_count}")


if __name__ == "__main__":
    asyncio.run(main())
