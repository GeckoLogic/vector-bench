from .base import BaseChunker


class FixedChunker(BaseChunker):
    """Split text into fixed-size token (word) chunks."""

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = max(1, chunk_size)

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def __repr__(self) -> str:
        return f"FixedChunker(chunk_size={self.chunk_size})"
