from .base import BaseChunker


class OverlappingChunker(BaseChunker):
    """Sliding window chunker with configurable overlap."""

    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = max(1, chunk_size)
        self.overlap = max(0, min(overlap, chunk_size - 1))

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = self.chunk_size - self.overlap
        if step < 1:
            step = 1
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
            i += step
        return chunks

    def __repr__(self) -> str:
        return f"OverlappingChunker(chunk_size={self.chunk_size}, overlap={self.overlap})"
