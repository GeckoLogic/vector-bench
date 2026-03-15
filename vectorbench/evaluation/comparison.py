import numpy as np
import pandas as pd
from .metrics import cosine_similarity, mean_pairwise_similarity


def compare_experiments(
    exp_a: dict,
    exp_b: dict,
    queries: list[str],
    query_embeddings_a: np.ndarray,
    query_embeddings_b: np.ndarray,
) -> pd.DataFrame:
    """Compare two experiments on a set of queries.

    exp_a, exp_b: dicts with keys 'config', 'chunks', 'embeddings', 'store'
    Returns DataFrame with per-query comparison.
    """
    rows = []
    store_a = exp_a["store"]
    store_b = exp_b["store"]

    for i, query in enumerate(queries):
        q_emb_a = query_embeddings_a[i]
        q_emb_b = query_embeddings_b[i]

        results_a = store_a.query(q_emb_a, n_results=1)
        results_b = store_b.query(q_emb_b, n_results=1)

        row = {
            "query": query,
            "top1_a": results_a[0]["document"] if results_a else "",
            "score_a": results_a[0]["score"] if results_a else 0.0,
            "top1_b": results_b[0]["document"] if results_b else "",
            "score_b": results_b[0]["score"] if results_b else 0.0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summary_stats(exp: dict) -> dict:
    """Compute summary statistics for an experiment."""
    embeddings = exp.get("embeddings")
    chunks = exp.get("chunks", [])
    if embeddings is None or len(embeddings) == 0:
        return {}
    return {
        "n_chunks": len(chunks),
        "mean_chunk_len": float(np.mean([len(c) for c in chunks])),
        "embedding_dim": embeddings.shape[1],
        "mean_pairwise_sim": mean_pairwise_similarity(embeddings),
    }
