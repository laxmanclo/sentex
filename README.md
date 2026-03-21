# sentex

**Sentence-graph context management for multi-agent AI pipelines.**

```
pip install sentex
```

---

The problem with multi-agent pipelines: by the time Agent 4 runs, it is carrying
the full output of Agents 1, 2, and 3. Every framework handles this the same way —
dump everything in, or make you manage it manually. Neither is good.

Sentex solves this with a sentence-level semantic graph. Every piece of context is
split into sentences, embedded, and connected by KNN edges across agent boundaries.
When an agent needs context, it submits a query and traverses the graph — collecting
exactly the sentences relevant to its task within its token budget. Nothing more.

---

## Quickstart

```python
from sentex import Pipeline, Read
import asyncio

pipeline = Pipeline()

@pipeline.agent(id="researcher", writes=["resources/search"])
async def researcher(ctx):
    # return value is automatically split, embedded, and added to the graph
    return your_search_function(ctx.query)

@pipeline.agent(
    id="writer",
    reads=[Read("resources/search", layer="l1", budget=2000)],
    writes=["working/script"],
    token_budget=4000,
)
async def writer(ctx):
    # ctx["resources/search"] → the 8 most relevant sentences from the graph
    # ctx.render() → formatted prompt block ready to send to an LLM
    return await ctx.llm(ctx.render())

result = asyncio.run(pipeline.run(
    query="explain how the immune system works",
    llm="gpt-4o",  # any LiteLLM-compatible model
))

print(result.outputs["working/script"])
print(result.summary())
```

---

## The four context levels

Every piece of content exists in four representations simultaneously:

| Level | What | Tokens | Used for |
|-------|------|--------|---------|
| **L0** | One-sentence identity | ~50 | Filtering — which nodes are worth loading |
| **L1** | Sentence-graph retrieval | budget-controlled | Precision retrieval via KNN BFS |
| **L2** | Coherent summary | ~2000 | Fallback when L1 confidence is low |
| **L3** | Full raw content | unlimited | When you need everything |

Declare which level you want per read. The assembler enforces your token budget and
falls back automatically (`l1 → l2 → l0`) if the declared level won't fit.

```python
Read("resources/search", layer="l1", budget=2000)  # sentence graph, 2k budget
Read("resources/query",  layer="l3")               # full raw content
Read("resources/docs",   layer="l2")               # summary
```

---

## How the sentence graph works

When content is ingested:

1. Split into sentences (NLTK Punkt tokenizer; code blocks and list items are atomic)
2. Each sentence is embedded (`all-MiniLM-L6-v2`, 384 dims, runs locally)
3. K nearest neighbours are computed across **all sentences in the graph** — not just
   the current node. Sentences from different agents get edges automatically.
4. L0 and L2 are generated async via LiteLLM (any model, non-blocking)

When an agent retrieves at L1:

```
query → embed → find highest-similarity entry point → BFS via KNN edges
      → collect sentences until token budget exhausted → return sorted by score
```

A 60-sentence output from Agent 2 might yield 8 sentences for Agent 4. Those 8 are
the ones that matter. Cross-agent: if Agent 1 wrote "T-cells recognize antigens" and
Agent 3 wrote "antigen recognition triggers the adaptive cascade", those two sentences
get an edge. Agent 4 asking about immune recognition traverses to both automatically.

---

## Agent manifests

Every agent declares what it reads and what it writes:

```python
@pipeline.agent(
    id="analyst",
    reads=[
        Read("resources/search", layer="l1", budget=2000),
        Read("working/notes",    layer="l2", budget=500),
    ],
    writes=["working/analysis"],
    token_budget=4000,
    fallback="l2",            # if l1 confidence < threshold, use l2
    confidence_threshold=0.5,
)
async def analyst(ctx):
    ...
```

The manifest enables:
- **Validation**: Sentex checks at run-start that every declared read will be produced
  before the agent runs. Fails fast with a clear error if not.
- **Assembly**: context is built from exactly the declared reads, nothing more.
- **Budget enforcement**: total assembled context is checked against `token_budget`
  before the agent runs. Never a runtime token overflow.

---

## Cross-run memory

With `persist=`, edge weights and node summaries survive between pipeline runs.
Each run improves retrieval for the next one — no explicit feedback required.

```python
pipeline = Pipeline(persist="./sentex.db")

# First run: baseline retrieval
result = await pipeline.run(query="immune systems", llm="gpt-4o")

# Second run: edges boosted by what was useful in run 1 → better retrieval
result = await pipeline.run(query="immune systems", llm="gpt-4o")

# See run history
print(pipeline.history())
```

The store also caches L0/L2 summaries so they are never regenerated on repeated content.

---

## Scopes

Node IDs support an optional scope prefix that mirrors the filesystem model:

| Scope | Purpose |
|-------|---------|
| `resources/` | Shared knowledge — search results, docs, reference data |
| `working/` | Ephemeral computation — agent outputs within this run |
| `memory/` | Cross-run learnings loaded from MemoryStore at run start |

```python
Read("resources/search", layer="l1", budget=2000)
# writes=["working/script"]
# writes=["memory/user-preferences"]   ← persisted and reloaded next run
```

Scopes are a naming convention, not a restriction — you can use flat node IDs too.

---

## HTTP server

Sentex ships a FastAPI server for TypeScript (or any HTTP client) pipelines:

```bash
uvicorn sentex.server:app --port 8765
```

```
POST /ingest        → ingest content into the graph
POST /assemble      → assemble context for an agent
POST /mark_used     → record which nodes were useful
GET  /nodes         → list all nodes with L0 summaries
GET  /health        → node count, sentence count
```

---

## What Sentex does not do

- No persistent agent memory across conversation sessions (pipeline context, not chat memory)
- No vector database (numpy handles the scale we care about)
- No graph database (Python dict handles adjacency)
- No cloud infrastructure (runs in-process; SQLite for persistence)
- No opinion on which LLM you use (any LiteLLM-compatible model for L0/L2 generation)

---

## Comparison

|  | LangChain | LlamaIndex | OpenViking | **Sentex** |
|--|-----------|------------|------------|---------|
| Sentence-level retrieval | No | No | No | **Yes** |
| Cross-agent graph edges | No | No | No | **Yes** |
| Declared agent dependencies | No | No | No | **Yes** |
| Budget enforcement pre-call | No | No | No | **Yes** |
| Requires vector DB | No | Yes | Yes | **No** |
| Requires graph DB | No | No | No | **No** |
| Cross-run memory | No | No | Yes | **Yes** |
| Works fully in-memory | Yes | No | No | **Yes** |
| Progressive level loading | No | Partial | Yes | **Yes** |

---

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `ENGRAM_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | Model for L0/L2 generation |
| `ENGRAM_TOKEN_MODEL` | `gpt-4o` | Tokenizer for budget counting |

---

## License

MIT
