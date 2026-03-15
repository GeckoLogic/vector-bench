from .base import BaseChunker


class DocumentChunker(BaseChunker):
    """Returns the entire document as a single chunk."""

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        return [text] if text else []

    def __repr__(self) -> str:
        return "DocumentChunker()"
