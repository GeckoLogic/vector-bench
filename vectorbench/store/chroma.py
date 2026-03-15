import numpy as np
import chromadb
from .base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "data/chroma_db",
        in_memory: bool = False,
    ):
        self.collection_name = collection_name
        self._persist_directory = persist_directory
        if in_memory:
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{"_idx": i} for i in range(len(ids))]
        else:
            # ChromaDB requires non-empty metadata dicts
            metadatas = [m if m else {"_idx": i} for i, m in enumerate(metadatas)]
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, count),
            include=["documents", "distances", "metadatas"],
        )
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB cosine distance: score = 1 - distance
            distance = results["distances"][0][i]
            score = 1.0 - distance
            output.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "score": score,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
        return output

    def get_all(self) -> dict:
        """Return all stored documents, embeddings, and metadatas."""
        result = self._collection.get(include=["embeddings", "documents", "metadatas"])
        embeddings = np.array(result["embeddings"], dtype=np.float32) if result["embeddings"] else np.empty((0, 0))
        return {
            "ids": result["ids"],
            "documents": result["documents"],
            "embeddings": embeddings,
            "metadatas": result["metadatas"],
        }

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
