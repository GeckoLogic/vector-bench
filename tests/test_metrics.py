import numpy as np
import pytest
from vectorbench.evaluation.metrics import (
    cosine_similarity, cosine_similarity_matrix,
    retrieval_rank, cluster_metrics, mean_pairwise_similarity
)


def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(a, b) + 1.0) < 1e-6


def test_cosine_matrix_shape():
    emb = np.random.rand(5, 10).astype(np.float32)
    mat = cosine_similarity_matrix(emb)
    assert mat.shape == (5, 5)


def test_cosine_matrix_diagonal():
    emb = np.random.rand(4, 10).astype(np.float32)
    mat = cosine_similarity_matrix(emb)
    np.testing.assert_allclose(np.diag(mat), np.ones(4), atol=1e-5)


def test_retrieval_rank_basic():
    # Query identical to first candidate → rank 1
    candidates = np.eye(5)
    query = candidates[0].copy()
    result = retrieval_rank(query, candidates, relevant_indices=[0])
    assert result["ranks"][0] == 1
    assert result["hit_at_1"] == 1.0


def test_mean_pairwise_similarity_single():
    emb = np.random.rand(1, 10).astype(np.float32)
    assert mean_pairwise_similarity(emb) == 1.0


def test_cluster_metrics_shape():
    emb = np.random.rand(10, 20).astype(np.float32)
    result = cluster_metrics(emb)
    assert "intra_sim" in result
    assert "silhouette" in result
