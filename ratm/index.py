"""Main indexing pipeline: AST extraction → embed → ChromaDB + BM25 → graph."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ratm.cache import ensure_dirs
from ratm.embed import embed_docs_batch
from ratm.vectorstore import VectorStore
from ratm.sparse import BM25Index


@dataclass
class IndexStats:
    n_files: int
    n_nodes: int
    n_edges: int
    n_chunks: int

    def __str__(self) -> str:
        return (
            f"Indexed {self.n_files} files | "
            f"{self.n_nodes} nodes | "
            f"{self.n_edges} edges | "
            f"{self.n_chunks} chunks"
        )


def _node_to_chunk_text(node: dict) -> str:
    """Convert an AST node to a plain-text chunk for embedding."""
    label = node.get("label", "")
    source_file = node.get("source_file", "")
    source_location = node.get("source_location", "")
    parts = [p for p in [label, source_file, source_location] if p]
    return "\n".join(parts)


def _node_to_metadata(node: dict) -> dict:
    """Extract ChromaDB-safe metadata from an AST node."""
    return {
        "node_id": str(node.get("id", "")),
        "source_file": str(node.get("source_file", "")),
        "source_location": str(node.get("source_location", "")),
        "label": str(node.get("label", "")),
        "file_type": str(node.get("file_type", "")),
    }


def index(
    root: Path,
    out_dir: Path = Path("ratm-out"),
    skip_summaries: bool = False,
    client=None,
) -> IndexStats:
    """Run the full indexing pipeline.

    Steps:
    1. Collect indexable code files
    2. Extract AST nodes + edges
    3. Dedupe nodes by id
    4. Build chunk texts from nodes
    5. Embed chunks (BGE-M3) → add to ChromaDB
    6. Build BM25 index over chunk texts
    7. Build NetworkX graph and run Leiden clustering
    8. Save graph JSON to ratm-out/graph.json

    Args:
        root: path to the repository or directory to index
        out_dir: where to write ratm-out/ artifacts
        skip_summaries: skip Claude summarization even if client provided
        client: optional Anthropic client; if provided and skip_summaries is False,
            generates chunk and community summaries
    """
    from graphify.extract import collect_files, extract
    from graphify.build import build
    from graphify.cluster import cluster

    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    ensure_dirs(out_dir)

    # 1. Collect files
    files = collect_files(root)
    if not files:
        return IndexStats(n_files=0, n_nodes=0, n_edges=0, n_chunks=0)

    # 2. Extract AST — extract() handles per-file caching internally
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extraction = extract(files, cache_root=root)

    raw_nodes: list[dict] = extraction.get("nodes", [])
    raw_edges: list[dict] = extraction.get("edges", [])

    # 3. Dedupe nodes by id (AST extractor can emit duplicates across files)
    seen_ids: set[str] = set()
    nodes: list[dict] = []
    for node in raw_nodes:
        nid = node.get("id")
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append(node)

    if not nodes:
        return IndexStats(n_files=len(files), n_nodes=0, n_edges=len(raw_edges), n_chunks=0)

    # 4. Build chunk texts and ids
    chunk_texts: list[str] = []
    chunk_ids: list[str] = []
    chunk_metadatas: list[dict] = []

    for node in nodes:
        text = _node_to_chunk_text(node)
        if not text.strip():
            continue
        chunk_texts.append(text)
        chunk_ids.append(str(node["id"]))
        chunk_metadatas.append(_node_to_metadata(node))

    n_chunks = len(chunk_texts)

    # 5. Embed + add to ChromaDB
    vectorstore = VectorStore(persist_dir=str(out_dir / "chroma"))
    vectorstore.reset()

    embeddings = embed_docs_batch(chunk_texts)
    _CHROMA_BATCH = 500
    for start in range(0, n_chunks, _CHROMA_BATCH):
        end = start + _CHROMA_BATCH
        vectorstore.add(
            ids=chunk_ids[start:end],
            embeddings=embeddings[start:end],
            documents=chunk_texts[start:end],
            metadatas=chunk_metadatas[start:end],
        )

    # 6. Build BM25 index
    bm25 = BM25Index()
    bm25.build(chunk_texts, chunk_ids)
    bm25.save(str(out_dir / "bm25"))

    if not skip_summaries and client is not None:
        from ratm.summarize import summarize_chunk
        node_by_id: dict[str, dict] = {str(n["id"]): n for n in nodes}
        chunk_summaries: dict[str, str] = {}
        for chunk_id in chunk_ids:
            node = node_by_id.get(chunk_id)
            if node is not None:
                chunk_summaries[chunk_id] = summarize_chunk(node, client, out_dir)
        (out_dir / "summaries.json").write_text(json.dumps(chunk_summaries), encoding="utf-8")

    # 7. Build NetworkX graph + cluster
    G: nx.Graph = build([extraction])
    try:
        _communities = cluster(G)
    except Exception:
        _communities = {}

    if not skip_summaries and client is not None and _communities:
        from ratm.summarize import summarize_community
        node_by_id_c: dict[str, dict] = {str(n["id"]): n for n in nodes}
        edge_list: list[dict] = raw_edges
        top_communities = sorted(_communities.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]
        community_summary_list: list[str] = []
        for _cid, node_ids in top_communities:
            c_nodes = [node_by_id_c[nid] for nid in node_ids if nid in node_by_id_c]
            c_node_id_set = set(node_ids)
            c_edges = [
                e for e in edge_list
                if str(e.get("source", "")) in c_node_id_set
                and str(e.get("target", "")) in c_node_id_set
            ]
            summary = summarize_community(c_nodes, c_edges, client, out_dir)
            community_summary_list.append(summary)
        (out_dir / "community_summaries.json").write_text(
            json.dumps(community_summary_list), encoding="utf-8"
        )

    # 8. Save graph JSON
    graph_data = nx.node_link_data(G, edges="edges")
    graph_path = out_dir / "graph.json"
    graph_path.write_text(json.dumps(graph_data), encoding="utf-8")

    return IndexStats(
        n_files=len(files),
        n_nodes=len(nodes),
        n_edges=len(raw_edges),
        n_chunks=n_chunks,
    )
