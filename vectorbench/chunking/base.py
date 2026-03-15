from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks. Returns list of non-empty strings."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
