from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts. Returns array of shape (n, dim)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return canonical model name."""
        ...
