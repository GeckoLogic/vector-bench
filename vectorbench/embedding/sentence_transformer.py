import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper around sentence-transformers SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        model = self._load()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        model = self._load()
        return model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name
