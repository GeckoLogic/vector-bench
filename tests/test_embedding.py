import numpy as np
import pytest
from tests.conftest import MockEmbedder


def test_mock_embedder_shape(mock_embedder, sample_text):
    texts = sample_text.split(".")[:5]
    texts = [t.strip() for t in texts if t.strip()]
    embeddings = mock_embedder.embed(texts)
    assert embeddings.shape == (len(texts), 384)


def test_mock_embedder_dtype(mock_embedder):
    emb = mock_embedder.embed(["hello"])
    assert emb.dtype == np.float32


def test_mock_embedder_dimension(mock_embedder):
    assert mock_embedder.dimension == 384


def test_mock_embedder_model_name(mock_embedder):
    assert mock_embedder.model_name == "mock-embedder"


def test_mock_embedder_batch_size(mock_embedder):
    texts = [f"text {i}" for i in range(100)]
    emb = mock_embedder.embed(texts, batch_size=16)
    assert emb.shape[0] == 100
