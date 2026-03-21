"""
End-to-end example: immune system pipeline using Sentex for context management.

Shows two usage patterns:

  Pattern A — Bring Your Own Agents (BYOA)
    Your agents run however you want. Sentex handles what gets passed between them.
    Three calls: put() → get() → used()

  Pattern B — Pipeline decorator (optional orchestration)
    Sentex owns the run loop. Agents are decorated async functions.

Run:
    python examples/immune_system_pipeline.py --pattern a --mock
    python examples/immune_system_pipeline.py --pattern b --mock

    OPENAI_API_KEY=sk-... python examples/immune_system_pipeline.py --pattern a
"""
import asyncio
import sys

# ------------------------------------------------------------------
# Fake agent functions (replace with real LangChain / CrewAI / etc.)
# ------------------------------------------------------------------

SEARCH_DATA = """
The immune system is composed of two main branches: innate and adaptive immunity.
Innate immunity provides immediate, non-specific defense against pathogens.
The adaptive immune system mounts targeted responses to specific antigens.

T-cells are lymphocytes that develop in the thymus and play a central role in adaptive immunity.
Cytotoxic T-cells (CD8+) directly kill infected or cancerous cells.
Helper T-cells (CD4+) coordinate the immune response by activating other immune cells.

B-cells produce antibodies that bind to specific antigens on pathogens.
When a B-cell encounters its matching antigen, it differentiates into plasma cells.
Plasma cells secrete large quantities of antibodies into the bloodstream.
Memory B-cells persist long after an infection and enable faster future responses.

Macrophages are phagocytic cells that engulf and digest cellular debris and pathogens.
They also present antigen fragments on MHC-II molecules to activate helper T-cells.
Natural killer (NK) cells destroy virus-infected cells without prior sensitisation.

Fever is triggered by pyrogens released by macrophages during infection.
Elevated body temperature inhibits bacterial replication and accelerates immune activity.
Vaccines work by training the immune system to recognise a pathogen without causing disease.
Autoimmune disorders occur when the immune system mistakenly attacks the body's own tissue.
"""


async def mock_llm(prompt: str) -> str:
    if "findings" in prompt.lower() or "analyst" in prompt.lower():
        return (
            "Key findings:\n"
            "- T-cells and B-cells coordinate the adaptive immune response.\n"
            "- Macrophages act as first responders and antigen presenters.\n"
            "- Fever is a systemic response that inhibits pathogen replication.\n"
            "- Memory cells enable faster responses to repeat infections."
        )
    return (
        "The immune system is a layered defense network. T-cells hunt infected "
        "cells while B-cells manufacture targeted antibodies. Macrophages patrol "
        "tissues and signal danger. Together they form an immediate innate shield "
        "followed by a precision adaptive strike that leaves lasting memory."
    )


# ==================================================================
# Pattern A — BYOA: put / get / used
# ==================================================================

async def run_byoa(use_mock: bool) -> None:
    from sentex import ContextGraph

    print("\nPattern A — Bring Your Own Agents")
    print("=" * 50)

    graph = ContextGraph()
    llm = mock_llm if use_mock else None   # replace with your agent/LLM

    # --- Agent 1: researcher ---
    # Your agent runs however it wants; you just store the output.
    print("[researcher] ingesting search data...")
    graph.put("resources/search", SEARCH_DATA, agent_id="researcher")

    # --- Agent 2: analyst ---
    # Ask Sentex for the relevant sentences from Agent 1's output.
    context = graph.get(
        "resources/search",
        query="key immune mechanisms and cell types",
        budget=1500,
    )
    print(f"[analyst] received {len(context)} sentences from graph")

    prompt = "Analyse these immune system facts:\n" + "\n".join(context)
    analysis = await llm(prompt) if use_mock else input("analyst prompt: ")

    graph.put("working/analysis", analysis, agent_id="analyst")
    graph.used("resources/search")  # boosts retrieval for future runs

    # --- Agent 3: writer ---
    # Gets sentences from both previous agents — graph edges cross automatically.
    search_ctx  = graph.get("resources/search",   query="write a video script", budget=1000)
    analysis_ctx = graph.get("working/analysis",  query="write a video script", budget=600)

    print(f"[writer] search sentences: {len(search_ctx)}, analysis: {len(analysis_ctx)} chars")

    prompt = (
        f"Research:\n{chr(10).join(search_ctx)}\n\n"
        f"Analysis:\n{analysis_ctx}\n\n"
        "Write a compelling 60-second video script about the immune system."
    )
    script = await llm(prompt) if use_mock else input("writer prompt: ")
    graph.put("working/script", script, agent_id="writer")

    print(f"\nGraph: {graph.node_count} nodes · {graph.sentence_count} sentences")
    print("\n--- Script ---")
    print(script)


# ==================================================================
# Pattern B — Pipeline decorator
# ==================================================================

async def run_pipeline(use_mock: bool) -> None:
    from sentex import Pipeline, Read
    from sentex.graph import ContextGraph
    from sentex.embedder import Embedder

    print("\nPattern B — Pipeline decorator")
    print("=" * 50)

    pipeline = Pipeline()

    @pipeline.agent(id="researcher", writes=["resources/search"])
    async def researcher(ctx):
        print(f"  [researcher] ingesting search data")
        return SEARCH_DATA

    @pipeline.agent(
        id="analyst",
        reads=[Read("resources/search", layer="l1", budget=1500)],
        writes=["working/analysis"],
        token_budget=3000,
    )
    async def analyst(ctx):
        sentences = ctx["resources/search"]
        print(f"  [analyst] {len(sentences)} sentences · {ctx.token_utilization:.0%} budget used")
        return await ctx.llm(f"Analyse:\n{ctx.render()}")

    @pipeline.agent(
        id="writer",
        reads=[
            Read("resources/search",  layer="l1", budget=1000),
            Read("working/analysis",  layer="l2", budget=600),
        ],
        writes=["working/script"],
        token_budget=4000,
    )
    async def writer(ctx):
        print(f"  [writer] layers: {ctx.assembled.layers_used}")
        return await ctx.llm(ctx.render() + "\n\nWrite a 60-second video script.")

    llm = mock_llm if use_mock else "gpt-4o"
    result = await pipeline.run(
        query="explain how the immune system works for a general audience",
        llm=llm,
    )

    print("\n" + result.summary())
    print("\n--- Script ---")
    print(result.outputs.get("working/script", "(no output)"))


# ==================================================================
# Entry point
# ==================================================================

if __name__ == "__main__":
    pattern = "a"
    use_mock = "--mock" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--pattern"):
            pattern = arg.split("=")[-1].strip() if "=" in arg else sys.argv[sys.argv.index(arg) + 1]

    if pattern == "b":
        asyncio.run(run_pipeline(use_mock))
    else:
        asyncio.run(run_byoa(use_mock))
