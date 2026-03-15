import hashlib
import numpy as np
from pathlib import Path
from typing import Callable

from ..chunking import (
    BaseChunker, FixedChunker, OverlappingChunker,
    SentenceChunker, ParagraphChunker, DocumentChunker
)
from ..embedding.base import BaseEmbedder
from ..store.chroma import ChromaVectorStore
from .config import ExperimentConfig


STRATEGY_MAP: dict[str, Callable[[ExperimentConfig], BaseChunker]] = {
    "fixed": lambda c: FixedChunker(chunk_size=c.chunk_size),
    "overlapping": lambda c: OverlappingChunker(chunk_size=c.chunk_size, overlap=c.chunk_overlap),
    "sentence": lambda c: SentenceChunker(sentences_per_chunk=c.sentences_per_chunk),
    "paragraph": lambda c: ParagraphChunker(min_length=c.paragraph_min_length),
    "document": lambda c: DocumentChunker(),
}


class ExperimentRunner:
    """End-to-end pipeline: chunk → embed → store."""

    def __init__(
        self,
        config: ExperimentConfig,
        embedder: BaseEmbedder | None = None,
        persist_directory: str = "data/chroma_db",
        in_memory: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        self.config = config
        self._embedder = embedder
        self.persist_directory = persist_directory
        self.in_memory = in_memory
        self.progress_callback = progress_callback

    def _get_embedder(self) -> BaseEmbedder:
        if self._embedder is not None:
            return self._embedder
        from ..embedding.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(self.config.model_name)

    def _get_chunker(self) -> BaseChunker:
        factory = STRATEGY_MAP.get(self.config.strategy)
        if factory is None:
            raise ValueError(f"Unknown strategy: {self.config.strategy}. Valid: {list(STRATEGY_MAP)}")
        return factory(self.config)

    def chunk(self, text: str) -> list[str]:
        return self._get_chunker().chunk(text)

    def run(self, text: str) -> dict:
        """Full pipeline. Returns result dict with chunks, embeddings, store."""
        # 1. Chunk
        chunks = self.chunk(text)
        if not chunks:
            raise ValueError("No chunks produced from text.")

        # 2. Embed with batched progress
        embedder = self._get_embedder()
        batch_size = 32
        all_embeddings = []
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            batch_emb = embedder.embed(batch, batch_size=batch_size)
            all_embeddings.append(batch_emb)
            if self.progress_callback:
                self.progress_callback(min(i + batch_size, total), total)
        embeddings = np.vstack(all_embeddings)

        # 3. Store
        store = ChromaVectorStore(
            collection_name=self.config.collection_name,
            persist_directory=self.persist_directory,
            in_memory=self.in_memory,
        )
        # Only add if collection is empty
        if store.count() == 0:
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {"chunk_index": i, "strategy": self.config.strategy, "model": self.config.model_name}
                for i in range(len(chunks))
            ]
            store.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "store": store,
            "config": self.config,
            "n_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
        }
