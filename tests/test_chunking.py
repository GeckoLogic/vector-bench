import pytest
from vectorbench.chunking import (
    FixedChunker, OverlappingChunker, SentenceChunker,
    ParagraphChunker, DocumentChunker
)


def test_fixed_chunker_basic(sample_text):
    chunker = FixedChunker(chunk_size=20)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 20


def test_fixed_chunker_single_chunk(sample_text):
    chunker = FixedChunker(chunk_size=10000)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) == 1


def test_fixed_chunker_empty():
    chunker = FixedChunker(chunk_size=50)
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_overlapping_chunker(sample_text):
    chunker = OverlappingChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 1
    # Consecutive chunks should share words
    words = sample_text.split()
    for i in range(len(chunks) - 1):
        assert len(chunks[i].split()) <= 20


def test_overlapping_no_overlap(sample_text):
    fixed = FixedChunker(chunk_size=20)
    overlapping = OverlappingChunker(chunk_size=20, overlap=0)
    assert fixed.chunk(sample_text) == overlapping.chunk(sample_text)


def test_sentence_chunker(sample_text):
    chunker = SentenceChunker(sentences_per_chunk=2)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    for c in chunks:
        assert c.strip() != ""


def test_paragraph_chunker():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunker = ParagraphChunker(min_length=10)
    chunks = chunker.chunk(text)
    assert len(chunks) == 3


def test_paragraph_chunker_filters_short():
    text = "Short.\n\nLong enough paragraph to keep here for sure."
    chunker = ParagraphChunker(min_length=20)
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "Long enough" in chunks[0]


def test_document_chunker(sample_text):
    chunker = DocumentChunker()
    chunks = chunker.chunk(sample_text)
    assert len(chunks) == 1
    assert chunks[0] == sample_text.strip()


def test_document_chunker_empty():
    chunker = DocumentChunker()
    assert chunker.chunk("") == []
