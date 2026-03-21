# sentex

**Context management middleware for multi-agent AI pipelines.**

```
pip install sentex
```

You have agents. They produce outputs. The next agent needs to read those outputs — but not all of them. Just the parts that are relevant to *its specific task*, within a token budget.

Sentex is the layer between your agents that makes this work.

---

## The problem

In any multi-agent pipeline, Agent 3 ends up carrying the full output of Agents 1 and 2 — whether it's relevant or not. At scale this blows token budgets and degrades output quality. Every framework either dumps everything in or makes you manage it manually.

## How Sentex works

Every agent output goes into a shared sentence graph. When the next agent runs, Sentex retrieves exactly the sentences relevant to its task — traversing semantic KNN edges across agent boundaries — and delivers them within a token budget you set.

```
Agent 1 output → split into sentences → embedded → KNN graph
                                                         ↓
Agent 3 query → find entry point → BFS traversal → 8 sentences (not 60)
```

---

## Quickstart — bring your own agents

```python
from sentex import ContextGraph

graph = ContextGraph()

# === Your Agent 1 runs (any framework — LangChain, CrewAI, raw API) ===
output = my_agent_1.run("research the immune system")
graph.put("search-results", output, agent_id="researcher")

# === Build context for Agent 2 ===
context = graph.get("search-results", query="write a 60-second script", budget=2000)
# context = [most relevant sentences from Agent 1's output]

prompt = f"Use this research:\n{chr(10).join(context)}\n\nWrite the script."
script = my_agent_2.run(prompt)

graph.put("script", script, agent_id="writer")
graph.used("search-results")   # boosts retrieval for future runs
```

Three calls: `put()`, `get()`, `used()`. That's the integration.

---

## The four context levels

Every node in the graph has four representations:

| Level | What | When to use |
|-------|------|-------------|
| **L0** | ~50-token identity sentence | Scanning many nodes to decide which to load |
| **L1** | Sentence-graph retrieval | Default — get the relevant sentences, nothing else |
| **L2** | ~300-token extractive summary | When you need a coherent overview, not fragments |
| **L3** | Full raw content | When the agent needs everything |

```python
# L1 — sentence graph (default)
context = graph.get("search-results", query="immune cells", budget=2000)
# → ["T-cells recognize antigens on pathogen surfaces.",
#    "Antigen recognition triggers the adaptive immune cascade.", ...]

# L2 — extractive summary (first ~300 tokens of content)
summary = graph.get("search-results", query="", budget=2000, layer="l2")
# → "The immune system has two branches: innate and adaptive. T-cells..."

# L3 — full content
full = graph.get("search-results", query="", budget=99999, layer="l3")
# → the entire raw output from Agent 1

# L0 — identity (first sentence, used by scan_nodes)
identity = graph.get("search-results", query="", budget=100, layer="l0")
# → "The immune system has two branches: innate and adaptive."
```

---

## Multi-agent example

```python
from sentex import ContextGraph

graph = ContextGraph()

# Agent 1: researcher
research = researcher_agent.run("immune system")
graph.put("resources/research", research, agent_id="researcher")

# Agent 2: analyst — only sees the research sentences relevant to its task
context = graph.get("resources/research", query="key mechanisms and findings", budget=1500)
analysis = analyst_agent.run(f"Analyse:\n{chr(10).join(context)}")
graph.put("working/analysis", analysis, agent_id="analyst")
graph.used("resources/research")

# Agent 3: writer — graph edges cross agent boundaries automatically.
# Sentences from research and analysis that are semantically close get KNN edges.
# BFS traversal for each node surfaces the most relevant from both.
research_ctx  = graph.get("resources/research", query="write a script", budget=1000)
analysis_ctx  = graph.get("working/analysis",   query="write a script", budget=800)

script = writer_agent.run(
    f"Research:\n{chr(10).join(research_ctx)}\n\n"
    f"Analysis:\n{analysis_ctx}\n\n"
    f"Write a 60-second video script."
)
graph.put("working/script", script, agent_id="writer")
```

---

## Structured assembly (optional)

If you want Sentex to handle budget enforcement across multiple reads automatically:

```python
from sentex import ContextGraph, defineAgent, Read

graph = ContextGraph()

# ... agents 1 and 2 have run, graph has content ...

writer_manifest = defineAgent(
    id="writer",
    reads=[
        Read("resources/research", layer="l1", budget=1500),
        Read("working/analysis",   layer="l2", budget=500),
    ],
    writes=["working/script"],
    token_budget=4000,      # total cap across all reads
    fallback="l2",          # if L1 confidence < 0.5, serve L2 instead
)

assembled = graph.assemble_for(writer_manifest, query="write a video script")

# assembled.context         → {"resources/research": [sentences], "working/analysis": "summary"}
# assembled.token_count     → 1923   (enforced before the LLM call — never a surprise)
# assembled.layers_used     → {"resources/research": "l1", "working/analysis": "l2"}
# assembled.confidence      → {"resources/research": 0.84, "working/analysis": 1.0}
# assembled.compressed      → []   (nothing fell back due to budget)
# assembled.missing         → []   (all declared reads were available)

# Build a prompt from the assembled context:
sections = []
for node_id, content in assembled.context.items():
    text = "\n".join(content) if isinstance(content, list) else content
    sections.append(f"[{node_id}]\n{text}")
prompt = "\n\n".join(sections) + "\n\nWrite a 60-second video script."

script = writer_agent.run(prompt)
graph.put("working/script", script, agent_id="writer")
graph.mark_used(assembled, used_ids=["resources/research"])
```

`assemble_for()` handles:
- Token budget across all reads (total cap, not per-read)
- Automatic fallback: L1 → L2 → L0 if over budget
- Confidence-based fallback: low similarity → serve L2 instead of bad sentences
- Full diagnostics on what was actually served

---

## Dynamic node discovery (AutoRead)

When you don't know the node IDs at definition time — scan all nodes at L0, retrieve from the top-k:

```python
from sentex import ContextGraph, AutoRead, defineAgent

graph = ContextGraph()

# ... many nodes ingested under resources/* ...

manifest = defineAgent(
    id="synthesiser",
    reads=[AutoRead(top_k=3, layer="l1", budget_per_node=1000, scope="resources")],
    writes=["working/synthesis"],
    token_budget=5000,
)
assembled = graph.assemble_for(manifest, query="immune system mechanisms")
# → scans all resources/* nodes at L0
# → retrieves L1 from the 3 most relevant
# → keys in assembled.context: "auto:resources/research", "auto:resources/docs", ...
```

---

## Graph inspection

```python
graph.stats()
# → {"nodes": 3, "sentences": 47, "edges": 235, "edge_boosts": 12, "node_ids": [...]}

graph.get_node("resources/research")
# → ContextNode(id=..., produced_by=..., sentence_ids=..., l0=..., l2=..., ...)

# Scan nodes by relevance (L0 level — fast, no sentence retrieval)
graph.scan_nodes("immune cell coordination", top_k=3)
# → [("resources/research", 0.87), ("working/analysis", 0.74), ...]
```

---

## HTTP server (for TypeScript / non-Python pipelines)

```bash
uvicorn sentex.server:app --port 8765
```

```
POST /put        body: {node_id, content, agent_id}
POST /get        body: {node_id, query, budget, layer}
POST /used       body: {node_ids: [...]}
POST /scan       body: {query, top_k, scope}
POST /assemble   body: {agent_id, reads, token_budget, query}
GET  /nodes      list all nodes with token counts
GET  /nodes/{id} inspect a single node
GET  /health     node count, sentence count
```

OpenAPI docs at `http://localhost:8765/docs`.

---

## Pipeline decorator (optional orchestration)

If you want Sentex to own the run loop rather than just managing context:

```python
from sentex import Pipeline, Read

pipeline = Pipeline()

@pipeline.agent(id="researcher", writes=["resources/research"])
async def researcher(ctx):
    return await ctx.llm("Research the immune system in detail.")

@pipeline.agent(
    id="writer",
    reads=[Read("resources/research", layer="l1", budget=2000)],
    writes=["working/script"],
    token_budget=4000,
)
async def writer(ctx):
    return await ctx.llm(ctx.render() + "\n\nWrite a 60-second video script.")

result = await pipeline.run(
    query="explain how the immune system works",
    llm="gpt-4o",   # any LiteLLM model string
)

print(result.outputs["working/script"])
print(result.summary())
```

---

## Cross-run memory (SQLite-backed)

Edge weights and node summaries persist between runs so retrieval improves over time:

```python
from sentex import Pipeline, MemoryStore

store = MemoryStore("./sentex.db")
pipeline = Pipeline(persist="./sentex.db")

# Run 1: baseline retrieval
# Run 2: L2 summaries cached, no regeneration needed
# Run 3+: session history available

store.all_sessions()
# → [{"session_id": ..., "query": ..., "started_at": ..., "committed_at": ...}, ...]
```

---

## How the sentence graph works

When content is ingested:
1. Split into sentences (NLTK Punkt; code blocks and lists are atomic)
2. Each sentence embedded (`all-MiniLM-L6-v2`, 384 dims, runs locally, no API key needed)
3. K nearest neighbours computed across **all sentences in the graph** — not just within the current node. Sentences from different agents get edges automatically.
4. Extractive L0 (first sentence) and L2 (first ~300 tokens) built immediately; LLM-generated summaries are optional and async.

When an agent retrieves at L1:
```
embed(query) → highest-sim sentence in this node (entry point)
             → BFS via KNN edges (crosses agent boundaries)
             → collect until token budget hit
             → return sorted by relevance
```

---

## What Sentex does not do

- No agent orchestration — bring your own (LangChain, CrewAI, AutoGen, anything)
- No vector database required (numpy handles the KNN)
- No graph database required (Python dict handles adjacency)
- No cloud infrastructure (runs in-process; SQLite for optional persistence)
- No opinion on which LLM you use (any LiteLLM-compatible model for L0/L2 generation)

---

## Comparison

| | LangChain | LlamaIndex | **Sentex** |
|--|-----------|------------|------------|
| Sentence-level retrieval | No | No | **Yes** |
| Cross-agent graph edges | No | No | **Yes** |
| Works with any agent framework | Yes | Yes | **Yes** |
| Budget enforcement pre-call | No | No | **Yes** |
| Requires vector DB | No | Yes | **No** |
| Runs fully in-memory | Yes | No | **Yes** |
| HTTP server for polyglot pipelines | No | No | **Yes** |

---

## License

MIT
