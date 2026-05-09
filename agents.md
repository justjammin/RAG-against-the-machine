# ratm — Agent Reference

Quick reference for agents working with RAG-against-the-machine.

## Quick start

```bash
# Build index (one time)
ratm index /path/to/repo

# Ask questions
ratm query "How does feature X work?"

# Check coverage
ratm status

# Wire MCP (optional)
ratm serve --out ratm-out
# Add to claude_desktop_config.json (see below)
```

## When to use ratm vs reading files

| Scenario | Use ratm? | Why |
|----------|-----------|-----|
| Small repo (<10 files) | No | Reading is faster, no index needed |
| Single file | No | Direct grep or read |
| Cross-file question ("How does X relate to Y?") | Yes | Can't answer without context |
| Large repo (50+ files) | Yes | Won't fit in context window |
| One-off question | No | Index cost not justified |
| Repeated queries (5+ questions) | Yes | Prompt caching amortizes index cost |

## MCP integration

Add to `~/.config/Claude/claude_desktop_config.json`:

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

Restart Claude. Now you have three tools:
- `query(question, top_k=10)` → answer + citations
- `index_status()` → coverage
- `graph_stats()` → structure

Prefer MCP tools if available (faster, returns structured JSON).

## Interpreting results

Every answer includes source citations:

```json
{
  "answer": "The system uses JWT tokens...",
  "chunks": [
    {
      "text": "function validateToken(...) { ... }",
      "source_file": "auth/session.py",
      "source_location": "validateToken",
      "rerank_score": 0.89,
      "metadata": {...}
    }
  ]
}
```

**Rerank score** (0-1): confidence that this chunk answers the question. Higher = more relevant. Use it to rank chunks by quality.

**Source file + location**: for verification. If answer needs more detail, read these files directly.

## Cost notes

- **Index time**: ~2000 tokens with `--summarize`, ~100 without
- **Query 1**: ~50 tokens (no cache)
- **Queries 2-10**: ~15 tokens each (prompt caching enabled)
- **HyDE**: one Haiku call per query (~20 tokens)

Use `--summarize` if querying 3+ times. Break-even at query 4-5.

## Typical workflow

```bash
# Session 1: Index repo
ratm index /Users/jammin/Documents/GitHub/myrepo --summarize

# Ask 5 questions (cheap after first, due to caching)
ratm query "Authentication flow?"
ratm query "How is caching implemented?"
ratm query "What's the database schema?"
ratm query "How do migrations work?"
ratm query "Error handling strategy?"

# Session 2 (later): Same repo, no re-index needed
# Queries still cheap — cache persists
ratm query "New question about the same codebase"
```

## Commands at a glance

```bash
ratm index <path> [--summarize] [--out dir]
ratm query "<q>" [--top-k N] [--out dir]
ratm status [--out dir]
ratm serve [--out dir]
```

See `ratm/skill.md` for full details.
