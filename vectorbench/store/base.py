from abc import ABC, abstractmethod
import numpy as np


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Add embeddings with documents and optional metadata."""
        ...

    @abstractmethod
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> list[dict]:
        """Query for nearest neighbors. Returns list of {id, document, score, metadata}."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return number of stored vectors."""
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete and reset the collection."""
        ...
