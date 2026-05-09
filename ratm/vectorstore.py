"""ChromaDB persistent vector store wrapper."""
from __future__ import annotations

from pathlib import Path


COLLECTION_NAME = "ratm_chunks"


class VectorStore:
    def __init__(self, persist_dir: str = "ratm-out/chroma") -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add or update chunks in the collection."""
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, embedding: list[float], n_results: int = 50) -> list[dict]:
        """Dense similarity search. Returns list of result dicts."""
        count = self._collection.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )
        output: list[dict] = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        """Number of chunks stored."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get(self, ids: list[str]) -> list[dict]:
        """Fetch specific chunks by ID. Returns list of {"id", "document", "metadata"}."""
        if not ids:
            return []
        result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        output: list[dict] = []
        for i in range(len(result["ids"])):
            output.append({
                "id": result["ids"][i],
                "document": result["documents"][i],
                "metadata": result["metadatas"][i],
            })
        return output

    def ids(self) -> list[str]:
        """All stored chunk IDs."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=[])
        return result["ids"]
