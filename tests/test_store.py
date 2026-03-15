import numpy as np
import pytest
from vectorbench.store.chroma import ChromaVectorStore


@pytest.fixture
def store():
    s = ChromaVectorStore("test_collection", in_memory=True)
    yield s
    s.delete_collection()


def test_add_and_count(store):
    emb = np.random.rand(5, 384).astype(np.float32)
    store.add(
        ids=[f"id_{i}" for i in range(5)],
        embeddings=emb,
        documents=[f"doc {i}" for i in range(5)],
    )
    assert store.count() == 5


def test_query_returns_results(store):
    emb = np.random.rand(10, 384).astype(np.float32)
    store.add(
        ids=[f"id_{i}" for i in range(10)],
        embeddings=emb,
        documents=[f"document number {i}" for i in range(10)],
    )
    q = np.random.rand(384).astype(np.float32)
    results = store.query(q, n_results=3)
    assert len(results) == 3
    for r in results:
        assert "id" in r
        assert "document" in r
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.5  # cosine can be slightly > 1 due to float precision


def test_delete_collection(store):
    emb = np.random.rand(3, 384).astype(np.float32)
    store.add(ids=["a", "b", "c"], embeddings=emb, documents=["x", "y", "z"])
    assert store.count() == 3
    store.delete_collection()
    assert store.count() == 0
