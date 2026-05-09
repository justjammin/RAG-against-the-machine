"""BGE cross-encoder reranker for candidate re-scoring."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_model: "CrossEncoder | None" = None


def _get_model() -> "CrossEncoder":
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 20) -> list[dict]:
    """Re-score candidates with cross-encoder, return top_k sorted descending."""
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c["document"]) for c in candidates]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    scored = [
        {**c, "rerank_score": float(scores[i])}
        for i, c in enumerate(candidates)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
