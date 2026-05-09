---
name: ratm
description: "retrieval-augmented code Q&A — index a repo, ask questions, get cited answers"
trigger: /ratm
---

# /ratm

Index a codebase once, then answer cross-file questions with cited sources. Uses hybrid search (dense + sparse + graph), reranking, and Claude for generation with prompt caching to keep repeated queries cheap.

## Usage

```
ratm index <path>                        # extract AST + embed chunks → ChromaDB + BM25
ratm index <path> --out ratm-out         # custom output directory
ratm index <path> --summarize            # add Claude chunk + community summaries (slower, better answers)

ratm query "<question>"                  # ask a question about the indexed codebase
ratm query "<question>" --out ratm-out   # use custom output directory
ratm query "<question>" --top-k 20       # return top 20 chunks instead of default 10

ratm status                              # show index statistics
ratm status --out ratm-out               # check custom output directory

ratm serve                               # start MCP server for agent access
ratm serve --out ratm-out                # use custom output directory
```

## What ratm is for

Use ratm when you need to ask a code question that spans multiple files or when the codebase won't fit in context. Three things it does better than reading files directly:

1. **Cross-file retrieval** - Find related code across a large codebase without reading every file
2. **Persistent index** - Index once, query unlimited times. Prompt caching makes repeated questions cheap (~10% of fresh LLM cost)
3. **Cited answers** - Every answer includes source locations (`source_file` + `source_location`) so you can verify or dig deeper

Use it for:
- Understanding how components relate (e.g., "how does authentication flow through the system?")
- Finding similar code patterns across modules
- Answering questions about large repos you're new to
- Repeated Q&A sessions on the same codebase (prompt caching saves tokens)

Skip ratm for:
- Single files (just read directly)
- Questions answerable by grepping one file
- One-off questions on small codebases (not worth indexing)

## Pipeline

**Index time:**
1. Collect all code files
2. Parse AST (tree-sitter) → extract functions, classes, relationships
3. Embed chunks with BGE-M3 (asymmetric: doc-space embeddings)
4. Store in ChromaDB (dense vector search)
5. Build BM25 index (keyword search)
6. Build graph from relationships
7. Optional: Claude summarize chunks + communities (with `--summarize`)

**Query time:**
1. HyDE: Claude generates hypothetical answer doc (bridges retriever vs generator gap)
2. Embed hypothetical + original query in doc-space
3. Dense retrieval (ChromaDB) → top-50
4. Sparse retrieval (BM25) → top-50
5. Graph-guided retrieval (PPR from seed nodes) → related chunks
6. Merge all candidates with RRF (reciprocal rank fusion)
7. Rerank with cross-encoder (BGE-reranker-v2-m3)
8. Assemble context: cached prefix (system + graph summary + community summaries) + dynamic chunks
9. Claude API → answer + citations

## Step-by-step: Index a codebase

### 1. First time - full index

```bash
ratm index /path/to/repo
```

Wait for extraction and embedding. Artifacts go to `ratm-out/`:
```
ratm-out/
├── chroma/              # ChromaDB vector store
├── bm25/                # BM25 sparse index
├── graph.json           # full relationship graph
└── cache/               # summary cache (if --summarize used)
```

### 2. With summaries for better answers

```bash
ratm index /path/to/repo --summarize
```

This costs Claude tokens at index time but dramatically improves answer quality:
- Chunks get one-line summaries
- Communities get synthetic summaries
- Answers include context (not just code snippets)
- Worth it if you'll query the repo repeatedly

### 3. Check what got indexed

```bash
ratm status
```

Output:
```
ratm-out: /path/to/ratm-out
  Vector store chunks: 245
  BM25 index: present (ratm-out/bm25)
  Graph: 312 nodes, 1047 edges
```

## Step-by-step: Query

### 1. Ask a question

```bash
ratm query "How does the authentication system work?"
```

Output format:
```
═══════════════ Answer ═══════════════
The system uses JWT tokens with a refresh flow. The AuthService generates 
tokens in validate_token(), stored in SessionCache. Middleware checks tokens 
on every request.

Sources (3 chunks):
  auth/session.py (validate_token)
  auth/middleware.py (TokenValidator)
  auth/cache.py (SessionCache.get_token)
```

### 2. Interpret citations

Each source includes:
- `source_file`: file path
- `source_location`: function/class name or line range
- `rerank_score`: how relevant this chunk is (0-1, higher = more relevant)

Use these to drill down:
```bash
# File mentions SessionCache, dig into that
ratm query "What does SessionCache do and how is it tested?"
```

### 3. Adjust retrieval

Return more chunks if the answer feels incomplete:
```bash
ratm query "Authentication flow" --top-k 20
```

Default is 10; more gives more context but can be noisy.

## MCP integration (for agents)

Wire ratm into Claude Desktop or other MCP clients so agents can access it live:

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ratm": {
      "command": "python3",
      "args": ["-m", "ratm.serve", "--out", "/absolute/path/to/ratm-out"]
    }
  }
}
```

Restart Claude Desktop. Agents now have three tools:
- `query(question: str, top_k?: int)` → list of cited chunks
- `index_status()` → current index size and types
- `graph_stats()` → node count, edge count, coverage

Start the server in a terminal:
```bash
ratm serve --out ratm-out
```

## MCP tools exposed

All tools return structured JSON with source citations.

### query
```python
query(question: str, top_k: int = 10) -> {
  "answer": str,
  "chunks": [
    {
      "text": str,
      "source_file": str,
      "source_location": str,
      "rerank_score": float,
      "metadata": dict
    }
  ]
}
```

### index_status
```python
index_status() -> {
  "out_dir": str,
  "chunks": int,
  "graph": {
    "nodes": int,
    "edges": int,
    "has_summaries": bool
  },
  "bm25": bool,
  "chroma": bool
}
```

### graph_stats
```python
graph_stats() -> {
  "total_nodes": int,
  "total_edges": int,
  "top_nodes": [{"id": str, "label": str, "degree": int}],
  "communities": int,
  "avg_degree": float
}
```

## When to use --summarize

Trade index-time tokens for better answer quality.

**Use --summarize if:**
- You'll query the repo 5+ times (prompt caching amortizes the cost)
- Answers need context, not just code snippets
- You care about cross-cutting concerns (architecture, design patterns)

**Skip --summarize if:**
- One-off indexing
- Only searching for specific functions
- Repo already well-documented
- Token budget is tight

Example cost:
- Index without summaries: ~5 min, ~100 tokens
- Index with summaries: ~15 min, ~2000 tokens
- Query 1 (with caching): ~50 tokens
- Query 2-10 (cached): ~15 tokens each

Break-even: query 3-4x.

## Output structure

```
ratm-out/
├── cache/
│   ├── ast/             # AST extraction cache (content-hash keyed)
│   └── summaries/       # chunk + community summaries
├── chroma/              # ChromaDB persistent directory
│   └── chroma.db
├── bm25/                # BM25 sparse index
│   ├── inverted.pkl
│   ├── idf.pkl
│   ├── corpus.pkl
│   └── ids.json         # id → filename mapping
├── graph.json           # full networkx graph (node_link format)
│   # {"nodes": [...], "edges": [...]}
├── GRAPH_REPORT.md      # structural analysis (if --summarize used)
└── community_summaries.json  # list of community descriptions (if --summarize)
```

Use `ratm-out/graph.json` to:
- Understand repo structure at a glance
- Find communities of related nodes
- Seed graph-guided retrieval (PPR)

## Output directory cleanup

If you want to rebuild from scratch:
```bash
rm -rf ratm-out/
ratm index /path/to/repo
```

To add new files to an existing index:
```bash
# Just run index again - detects changes, re-extracts + merges
ratm index /path/to/repo
```

To use a different output directory:
```bash
ratm index /path/to/repo --out my-custom-out
ratm query "question" --out my-custom-out
```

All three commands (`index`, `query`, `status`, `serve`) default to `ratm-out` but respect `--out`.
