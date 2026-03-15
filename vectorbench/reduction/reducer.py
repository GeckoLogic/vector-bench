import numpy as np
from typing import Literal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def reduce_embeddings(
    embeddings: np.ndarray,
    method: Literal["umap", "tsne", "pca"] = "umap",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D or 3D.

    Returns array of shape (n, n_components).
    Falls back to PCA if umap-learn is not installed.
    """
    n = embeddings.shape[0]

    if n <= n_components:
        # Pad with zeros if too few samples
        result = np.zeros((n, n_components))
        result[:, :min(n, n_components)] = embeddings[:, :min(embeddings.shape[1], n_components)]
        return result

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, n - 1),
                min_dist=min_dist,
                random_state=random_state,
                verbose=False,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            method = "pca"

    if method == "tsne":
        perplexity = min(30, n - 1)
        reducer = TSNE(
            n_components=min(n_components, 3),
            perplexity=perplexity,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    # PCA fallback
    reducer = PCA(n_components=min(n_components, embeddings.shape[1], n))
    return reducer.fit_transform(embeddings)
