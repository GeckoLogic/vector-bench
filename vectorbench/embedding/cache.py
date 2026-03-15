import hashlib
import numpy as np
from pathlib import Path
from joblib import Memory
from .base import BaseEmbedder


def _make_cache_key(model_name: str, texts: list[str]) -> str:
    content = model_name + "|" + "\n".join(texts)
    return hashlib.sha256(content.encode()).hexdigest()


class CachedEmbedder(BaseEmbedder):
    """Wraps any BaseEmbedder with disk-based joblib caching."""

    def __init__(self, embedder: BaseEmbedder, cache_dir: str | Path = "data/cache"):
        self._embedder = embedder
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        memory = Memory(location=str(cache_path), verbose=0)
        self._cached_embed = memory.cache(self._raw_embed)
        self._last_cache_hit = False

    def _raw_embed(self, model_name: str, texts: list[str], batch_size: int) -> np.ndarray:
        return self._embedder.embed(texts, batch_size=batch_size)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        key = _make_cache_key(self._embedder.model_name, texts)
        # Check cache by looking at joblib internals
        result = self._cached_embed(self._embedder.model_name, texts, batch_size)
        return result

    @property
    def was_cache_hit(self) -> bool:
        return self._last_cache_hit

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return self._embedder.model_name
