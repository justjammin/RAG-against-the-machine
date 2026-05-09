"""BGE-M3 embedding wrapper using sentence-transformers. Lazy-loads model on first call."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "Represent this question for searching relevant code: "
DOC_PREFIX = "Represent this code/document for retrieval: "
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
_BATCH_SIZE = 32

_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(text: str) -> list[float]:
    """Embed a query string with the asymmetric query prefix."""
    model = _get_model()
    vec = model.encode(QUERY_PREFIX + text, normalize_embeddings=True)
    return vec.tolist()


def embed_doc(text: str) -> list[float]:
    """Embed a document string with the asymmetric document prefix."""
    model = _get_model()
    vec = model.encode(DOC_PREFIX + text, normalize_embeddings=True)
    return vec.tolist()


def embed_docs_batch(texts: list[str]) -> list[list[float]]:
    """Batch-encode a list of document strings with DOC_PREFIX."""
    model = _get_model()
    prefixed = [DOC_PREFIX + t for t in texts]
    vecs = model.encode(prefixed, batch_size=_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_raw(text: str) -> list[float]:
    """Embed without any prefix — for HyDE hypothetical documents."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()
