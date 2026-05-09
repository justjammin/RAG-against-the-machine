"""Claude-powered summarization for graph nodes and communities. Cached to disk."""
from __future__ import annotations

from pathlib import Path

from ratm.cache import content_hash, load_summary, save_summary


def summarize_chunk(
    node: dict,
    client,
    out_dir: Path,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Return a one-sentence summary of what this code symbol does.

    Cache keyed by content hash of label + source_file + source_location.
    Returns empty string on any Claude error.
    """
    label = node.get("label", "")
    source_file = node.get("source_file", "")
    source_location = node.get("source_location", "")
    text = "\n".join(p for p in [label, source_file, source_location] if p)
    h = content_hash(text)

    cached = load_summary(h, out_dir)
    if cached is not None:
        return cached

    try:
        response = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"In one sentence, describe what this code symbol does.\n"
                        f"Symbol: {label}\n"
                        f"File: {source_file}"
                    ),
                }
            ],
        )
        summary = response.content[0].text.strip()
    except Exception:
        return ""

    save_summary(h, summary, out_dir)
    return summary


def summarize_community(
    community_nodes: list[dict],
    edges: list[dict],
    client,
    out_dir: Path,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Return a summary of what this cluster of code symbols does together.

    Cache keyed by hash of sorted node ids.
    Returns empty string on any Claude error.
    """
    node_ids = sorted(str(n.get("id", "")) for n in community_nodes)
    h = content_hash("\n".join(node_ids))

    cached = load_summary(h, out_dir)
    if cached is not None:
        return cached

    labels = [n.get("label", "") for n in community_nodes if n.get("label")]
    edge_pairs = [
        f"{e.get('source', '')} → {e.get('target', '')}"
        for e in edges
        if e.get("source") and e.get("target")
    ]

    symbols_text = ", ".join(labels[:40])
    edges_text = "; ".join(edge_pairs[:20]) if edge_pairs else "none"

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize what this cluster of {len(community_nodes)} code symbols does together.\n"
                        f"Symbols: {symbols_text}\n"
                        f"Relationships: {edges_text}"
                    ),
                }
            ],
        )
        summary = response.content[0].text.strip()
    except Exception:
        return ""

    save_summary(h, summary, out_dir)
    return summary
