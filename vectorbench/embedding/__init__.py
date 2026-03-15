from .base import BaseEmbedder
from .registry import MODEL_REGISTRY
from .cache import CachedEmbedder

__all__ = ["BaseEmbedder", "SentenceTransformerEmbedder", "MODEL_REGISTRY", "CachedEmbedder"]


def __getattr__(name):
    if name == "SentenceTransformerEmbedder":
        from .sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
