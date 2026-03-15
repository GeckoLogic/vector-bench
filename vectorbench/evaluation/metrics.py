import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """NxN cosine similarity matrix."""
    normed = normalize(embeddings, norm="l2")
    return normed @ normed.T


def retrieval_rank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    relevant_indices: list[int],
) -> dict:
    """Compute retrieval metrics for a single query."""
    sims = np.array([cosine_similarity(query_embedding, c) for c in candidate_embeddings])
    ranked = np.argsort(-sims)
    ranks = {idx: int(np.where(ranked == idx)[0][0]) + 1 for idx in relevant_indices}
    mrr = sum(1.0 / r for r in ranks.values()) / len(ranks) if ranks else 0.0
    hits_at_1 = sum(1 for r in ranks.values() if r <= 1)
    hits_at_5 = sum(1 for r in ranks.values() if r <= 5)
    return {
        "ranks": ranks,
        "mrr": mrr,
        "hit_at_1": hits_at_1 / len(relevant_indices),
        "hit_at_5": hits_at_5 / len(relevant_indices),
        "mean_rank": float(np.mean(list(ranks.values()))),
    }


def cluster_metrics(embeddings: np.ndarray, labels: list[int] | None = None) -> dict:
    """Compute cluster quality metrics."""
    if embeddings.shape[0] < 2:
        return {"silhouette": None, "intra_sim": None, "inter_sim": None}
    sim_matrix = cosine_similarity_matrix(embeddings)
    n = len(embeddings)
    mask = ~np.eye(n, dtype=bool)
    intra_sim = float(sim_matrix[mask].mean())
    result = {"intra_sim": intra_sim, "inter_sim": None, "silhouette": None}
    if labels is not None and len(set(labels)) > 1:
        try:
            result["silhouette"] = float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception:
            pass
    return result


def mean_pairwise_similarity(embeddings: np.ndarray) -> float:
    """Mean cosine similarity of all pairs."""
    if embeddings.shape[0] < 2:
        return 1.0
    sim = cosine_similarity_matrix(embeddings)
    n = len(embeddings)
    mask = ~np.eye(n, dtype=bool)
    return float(sim[mask].mean())
