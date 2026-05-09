# Project: RAG-against-the-machine

## What is ratm?

ratm indexes code and answers questions with cited sources. Index once, query unlimited times with prompt caching for cheap repeated access.

## When to use ratm

Use `ratm query` instead of reading files if:
- **Cross-file questions** — "How do X and Y relate?" — spans multiple files
- **Large codebase** — wouldn't fit in context (50+ files)
- **Repeated Q&A** — same repo, many questions (prompt caching saves tokens)

Use direct file reading if:
- Single file or tiny module
- Answer obvious from grepping one file
- One-off question on small corpus

## Quick setup

After `pip install -e .`, install the Claude Code skill (if `~/.claude/skills/ratm/SKILL.md` is missing):
```bash
ratm install
```

Check if index exists:
```bash
ratm status
```

Build index:
```bash
ratm index /path/to/code
```

Query:
```bash
ratm query "How does [system] work?"
```

Full command reference: `ratm/skill.md`

## MCP server access

If `ratm serve` is running (wired in `claude_desktop_config.json`), agents prefer MCP tools:
- `query(question, top_k)` — faster, structured
- `index_status()` — check coverage
- `graph_stats()` — repo structure

See `ratm/skill.md` for MCP integration details.
