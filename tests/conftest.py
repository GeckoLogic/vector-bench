import pytest
import numpy as np
from vectorbench.embedding.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder for tests — no model loading."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._model_name = "mock-embedder"

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        rng = np.random.default_rng(seed=sum(len(t) for t in texts))
        return rng.random((len(texts), self._dim)).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


@pytest.fixture
def sample_text() -> str:
    return (
        "Artificial intelligence is the simulation of human intelligence processes by machines. "
        "Machine learning is a subset of AI that enables systems to learn from data. "
        "Deep learning uses neural networks with many layers to model complex patterns. "
        "Natural language processing allows computers to understand and generate human language. "
        "Computer vision enables machines to interpret and understand visual information. "
        "Reinforcement learning trains agents through reward and penalty signals. "
        "Transfer learning applies knowledge from one domain to another related domain. "
        "Supervised learning trains models on labeled data to make predictions. "
        "Unsupervised learning finds hidden patterns in unlabeled data. "
        "Neural networks are inspired by the structure and function of the human brain."
    )


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def small_text() -> str:
    return "Hello world. This is a test. Short text for quick testing."
