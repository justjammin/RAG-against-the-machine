"""Hybrid retrieval: dense (ChromaDB) + BM25 + HyDE + PPR, fused with RRF then cross-encoder reranked."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future

from ratm.vectorstore import VectorStore
from ratm.sparse import BM25Index
from ratm.rerank import rerank
from ratm.graph import load_graph, ppr
from ratm.embed import embed_raw
from ratm.generate import hyde_expand

_GRAPH_CACHE: dict[str, tuple] = {}


def _cached_graph(graph_path: str) -> tuple:
    if graph_path not in _GRAPH_CACHE:
        _GRAPH_CACHE[graph_path] = load_graph(graph_path)
    return _GRAPH_CACHE[graph_path]


def rrf_merge(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Each list contains dicts with at least an "id" key.
    Returns merged list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)

    seen: dict[str, dict] = {}
    for ranked in ranked_lists:
        for item in ranked:
            if item["id"] not in seen:
                seen[item["id"]] = item

    merged = [
        {**seen[item_id], "score": score}
        for item_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return merged


def retrieve(
    query: str,
    vectorstore: VectorStore,
    bm25: BM25Index,
    embed_fn,
    n_dense: int = 50,
    n_bm25: int = 50,
    n_results: int = 20,
    graph_path: str | None = None,
    ppr_alpha: float = 0.85,
    hyde_client=None,
) -> list[dict]:
    """Hybrid retrieval: dense + BM25 + optional HyDE + optional PPR → RRF → rerank.

    embed_fn: callable that takes a query string and returns list[float].
    Returns list of dicts: {"id", "document", "metadata", "score", "rerank_score"}.
    """
    query_embedding = embed_fn(query)

    with ThreadPoolExecutor(max_workers=4) as executor:
        dense_future: Future = executor.submit(
            vectorstore.search, query_embedding, n_dense
        )
        bm25_future: Future = executor.submit(bm25.search, query, n_bm25)

        hyde_future: Future | None = None
        if hyde_client is not None:
            def _hyde_search():
                hyde_doc = hyde_expand(query, hyde_client)
                hyde_vec = embed_raw(hyde_doc)
                return vectorstore.search(hyde_vec, n_results=n_dense)
            hyde_future = executor.submit(_hyde_search)

        dense_raw = dense_future.result()

        ppr_future: Future | None = None
        if graph_path is not None:
            seed_ids = [r["id"] for r in dense_raw[:10]]
            def _ppr_search():
                A, id_to_idx, idx_to_id = _cached_graph(graph_path)
                return ppr(A, seed_ids, id_to_idx, idx_to_id, alpha=ppr_alpha, top_k=n_dense)
            ppr_future = executor.submit(_ppr_search)

        bm25_raw = bm25_future.result()
        hyde_raw = hyde_future.result() if hyde_future is not None else None
        ppr_raw = ppr_future.result() if ppr_future is not None else None

    dense_ranked = [
        {"id": r["id"], "document": r["document"], "metadata": r["metadata"], "score": 1.0 - r["distance"]}
        for r in dense_raw
    ]

    dense_lookup: dict[str, dict] = {r["id"]: r for r in dense_raw}

    bm25_only_ids = [item["id"] for item in bm25_raw if item["id"] not in dense_lookup]
    if bm25_only_ids:
        fetched = vectorstore.get(bm25_only_ids)
        for doc in fetched:
            dense_lookup[doc["id"]] = doc

    enriched_bm25: list[dict] = []
    for item in bm25_raw:
        meta_source = dense_lookup.get(item["id"], {})
        enriched_bm25.append({
            "id": item["id"],
            "document": meta_source.get("document", ""),
            "metadata": meta_source.get("metadata", {}),
            "score": item["score"],
        })

    ranked_lists: list[list[dict]] = [dense_ranked, enriched_bm25]

    if hyde_raw is not None:
        hyde_ranked = [
            {"id": r["id"], "document": r["document"], "metadata": r["metadata"], "score": 1.0 - r["distance"]}
            for r in hyde_raw
        ]
        ranked_lists.append(hyde_ranked)

    if ppr_raw is not None:
        ppr_lookup: dict[str, dict] = {r["id"]: r for r in dense_raw}
        ppr_ranked: list[dict] = []
        for item in ppr_raw:
            meta_source = ppr_lookup.get(item["id"], {})
            ppr_ranked.append({
                "id": item["id"],
                "document": meta_source.get("document", ""),
                "metadata": meta_source.get("metadata", {}),
                "score": item["score"],
            })
        ranked_lists.append(ppr_ranked)

    merged = rrf_merge(ranked_lists)
    return rerank(query, merged, top_k=n_results)
