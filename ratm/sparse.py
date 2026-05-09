"""BM25 index wrapper using bm25s library.

bm25s saves to a directory (not a single file). We persist a JSON sidecar
alongside the index directory that maps corpus positions → our chunk ids.
"""
from __future__ import annotations

import json
from pathlib import Path


_DEFAULT_INDEX_DIR = "ratm-out/bm25"
_IDS_SIDECAR = "ids.json"


class BM25Index:
    def __init__(self) -> None:
        self._retriever = None
        self._ids: list[str] = []

    def build(self, corpus: list[str], ids: list[str]) -> None:
        """Tokenize corpus and fit BM25. corpus[i] maps to ids[i]."""
        import bm25s
        self._ids = list(ids)
        self._retriever = bm25s.BM25()
        tokens = bm25s.tokenize(corpus, show_progress=False)
        self._retriever.index(tokens, show_progress=False)

    def search(self, query: str, n_results: int = 50) -> list[dict]:
        """Return top-n results as [{"id": str, "score": float}].

        Returns empty list if index not built or query yields no results.
        """
        if not self.is_built():
            return []
        import bm25s
        actual_k = min(n_results, len(self._ids))
        if actual_k == 0:
            return []
        query_tokens = bm25s.tokenize([query], show_progress=False)
        results, scores = self._retriever.retrieve(
            query_tokens, k=actual_k, show_progress=False
        )
        # results[0] is array of integer corpus indices
        output: list[dict] = []
        for i in range(len(results[0])):
            idx = int(results[0][i])
            output.append({"id": self._ids[idx], "score": float(scores[0][i])})
        return output

    def save(self, path: str = _DEFAULT_INDEX_DIR) -> None:
        """Persist index + id mapping to disk. path is a directory."""
        if not self.is_built():
            raise RuntimeError("BM25Index.save() called before build()")
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(save_dir), show_progress=False)
        sidecar = save_dir / _IDS_SIDECAR
        sidecar.write_text(json.dumps(self._ids), encoding="utf-8")

    def load(self, path: str = _DEFAULT_INDEX_DIR) -> None:
        """Load index + id mapping from disk."""
        import bm25s
        save_dir = Path(path)
        self._retriever = bm25s.BM25.load(str(save_dir), load_corpus=False)
        sidecar = save_dir / _IDS_SIDECAR
        self._ids = json.loads(sidecar.read_text(encoding="utf-8"))

    def is_built(self) -> bool:
        """True if the index has been built (or loaded)."""
        return self._retriever is not None and len(self._ids) > 0
