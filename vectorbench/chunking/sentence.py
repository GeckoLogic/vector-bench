import nltk
from .base import BaseChunker


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


class SentenceChunker(BaseChunker):
    """Split text on sentence boundaries using NLTK."""

    def __init__(self, sentences_per_chunk: int = 3):
        self.sentences_per_chunk = max(1, sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        _ensure_nltk()
        sentences = nltk.sent_tokenize(text.strip())
        chunks = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            group = sentences[i : i + self.sentences_per_chunk]
            chunk = " ".join(group).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def __repr__(self) -> str:
        return f"SentenceChunker(sentences_per_chunk={self.sentences_per_chunk})"
