"""Claude API generation with tiered prompt caching."""
from __future__ import annotations

import anthropic

SYSTEM_PROMPT = """\
You are a codebase assistant with deep knowledge of the indexed repository.
Answer questions about the code precisely and concisely. When referring to
code, include specific file paths and line numbers when available from the
provided context. If you are unsure, say so rather than guessing.
Cite sources from the retrieved code chunks in your answer.
"""


def generate_answer(
    query: str,
    chunks: list[dict],
    graph_summary: str,
    community_summaries: str,
    client: anthropic.Anthropic,
) -> str:
    """Generate an answer from Claude using tiered prompt caching.

    The system prompt, graph_summary, and community_summaries are marked
    with cache_control so repeated queries reuse the cached prefix.
    The retrieved chunks and the query itself are dynamic (no cache_control).

    Args:
        query: the user's question
        chunks: list of retrieved chunks, each with "document", "metadata" keys
        graph_summary: global graph structure summary text
        community_summaries: cluster/community summaries text
        client: initialized Anthropic client
    """
    chunks_text = _format_chunks(chunks)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"## Codebase Structure\n{graph_summary}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"## Module Summaries\n{community_summaries}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"## Relevant Code\n{chunks_text}",
                    },
                    {
                        "type": "text",
                        "text": f"Question: {query}\n\nAnswer with specific file paths and line numbers when available.",
                    },
                ],
            }
        ],
    )
    return response.content[0].text


def hyde_expand(query: str, client, model: str = "claude-haiku-4-5-20251001") -> str:
    """Generate a hypothetical document that would answer the query (HyDE)."""
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a short code excerpt or explanation using real identifiers "
                    f"that directly answers this question: {query}"
                ),
            }
        ],
    )
    return response.content[0].text


def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks as a readable context block."""
    if not chunks:
        return "(no relevant chunks retrieved)"
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source_file", "unknown")
        location = meta.get("source_location", "")
        label = meta.get("label", "")
        header = f"[{i}] {label} — {source}"
        if location:
            header += f" {location}"
        parts.append(f"{header}\n{chunk.get('document', '')}")
    return "\n\n".join(parts)
