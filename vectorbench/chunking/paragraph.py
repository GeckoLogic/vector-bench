import re
from .base import BaseChunker


class ParagraphChunker(BaseChunker):
    """Split text on double-newline paragraph boundaries."""

    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def chunk(self, text: str) -> list[str]:
        raw = re.split(r"\n\s*\n", text.strip())
        chunks = []
        for para in raw:
            para = para.strip()
            if len(para) >= self.min_length:
                chunks.append(para)
        return chunks

    def __repr__(self) -> str:
        return f"ParagraphChunker(min_length={self.min_length})"
