"""Page 6: UMAP/t-SNE embedding space explorer."""
import streamlit as st
import numpy as np
from app.state import _init_defaults, get_current_experiment
from app.components.scatter_plot import render_scatter_2d, render_scatter_3d
from vectorbench.reduction.reducer import reduce_embeddings

_init_defaults()
st.set_page_config(page_title="Explorer — VectorBench", layout="wide")
st.title("6. Embedding Space Explorer")
st.markdown(
    "High-dimensional embedding vectors are impossible to visualise directly, but dimensionality reduction "
    "techniques like UMAP, t-SNE, and PCA can project them into 2D or 3D space while preserving their "
    "relative structure. Each point on the chart is a chunk — hover over it to read the text. "
    "Semantically similar chunks will cluster together, revealing how the embedding model organises meaning. "
    "Overlay multiple experiments to compare how different models or chunking strategies shape the same content "
    "in the embedding space."
)

saved = st.session_state.get("saved_experiments", {})
if not saved:
    st.warning("No experiments available. Run the **Embed** step first.")
    st.stop()

exp_ids = list(saved.keys())
exp_options = {eid: saved[eid]["config"].experiment_name for eid in exp_ids}

selected_ids = st.multiselect(
    "Select Experiments to Visualize",
    exp_ids,
    default=exp_ids[:1],
    format_func=lambda x: exp_options[x],
    max_selections=4,
)

if not selected_ids:
    st.info("Select at least one experiment.")
    st.stop()

_METHOD_INFO = {
    "umap": (
        "**UMAP** (Uniform Manifold Approximation and Projection) preserves both local cluster structure "
        "and broader global relationships between groups. It is fast, scales well to large datasets, and "
        "typically produces the most interpretable layouts for embedding spaces. "
        "The two sliders below let you tune how the layout is computed."
    ),
    "tsne": (
        "**t-SNE** (t-distributed Stochastic Neighbor Embedding) is excellent at revealing tight local "
        "clusters — points that are nearest neighbours in high-dimensional space will be pulled together. "
        "However, it does *not* preserve global distances: the positions of clusters relative to each other "
        "are not meaningful, only the membership within a cluster is. Slower than UMAP on large datasets."
    ),
    "pca": (
        "**PCA** (Principal Component Analysis) is a linear method that projects data along the directions "
        "of greatest variance. It is fully deterministic (always produces the same result), extremely fast, "
        "and preserves global variance structure. Because it cannot capture non-linear relationships, "
        "clusters may appear less distinct than with UMAP or t-SNE, but the axes have a concrete meaning: "
        "distance along an axis corresponds to real variance in the embedding space."
    ),
}

col_method, col_nn, col_md = st.columns(3)
with col_method:
    method = st.radio("Reduction Method", ["umap", "tsne", "pca"])
with col_nn:
    n_neighbors = st.slider(
        "n_neighbors", 2, 50, 15,
        disabled=method != "umap",
        help=(
            "How many neighbouring points UMAP considers when learning local structure. "
            "Low values (2–5): fine local detail, many small tight clusters. "
            "High values (30–50): broader neighbourhood, more global structure, smoother layout."
        ),
    )
with col_md:
    min_dist = st.slider(
        "min_dist", 0.0, 1.0, 0.1,
        disabled=method != "umap",
        help=(
            "Minimum distance between points in the 2D/3D layout. "
            "Near 0: points pack tightly, clusters are compact and distinct. "
            "Higher values: points spread out more evenly, easier to read but less defined clusters."
        ),
    )

st.info(_METHOD_INFO[method])

# Gather embeddings from selected experiments
all_embeddings = []
all_labels = []
all_texts = []

for eid in selected_ids:
    exp = saved[eid]
    emb = exp["embeddings"]
    chunks = exp["chunks"]
    label = exp_options[eid]
    all_embeddings.append(emb)
    all_labels.extend([label] * len(chunks))
    all_texts.extend(chunks)

# Check if all experiments share the same embedding dimension
dims = [e.shape[1] for e in all_embeddings]
mixed_dims = len(set(dims)) > 1

n_components = st.radio("Dimensions", [2, 3], horizontal=True)

# Cache key
cache_key = (tuple(selected_ids), n_components, method, n_neighbors, min_dist)
reduction_cache = st.session_state.get("reduction_cache", {})

if cache_key not in reduction_cache:
    if mixed_dims:
        # Experiments have different dimensions (e.g. 384 vs 768) — reduce each independently
        # then concatenate the low-dimensional coordinates.
        st.info(
            f"Experiments use different embedding dimensions ({', '.join(str(d)+'d' for d in dims)}). "
            "Each is reduced independently — relative positions within an experiment are meaningful, "
            "but cross-experiment distances are not directly comparable."
        )
        total = sum(e.shape[0] for e in all_embeddings)
        with st.spinner(f"Running {method.upper()} per experiment ({total} points total)..."):
            coord_parts = [
                reduce_embeddings(
                    emb,
                    method=method,
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                )
                for emb in all_embeddings
            ]
        coords = np.vstack(coord_parts)
    else:
        combined = np.vstack(all_embeddings)
        with st.spinner(f"Running {method.upper()} ({combined.shape[0]} points)..."):
            coords = reduce_embeddings(
                combined,
                method=method,
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
    reduction_cache[cache_key] = coords
    st.session_state.reduction_cache = reduction_cache
else:
    coords = reduction_cache[cache_key]
    st.caption("Using cached reduction.")

if n_components == 2:
    render_scatter_2d(coords, all_labels, all_texts, title=f"{method.upper()} — {', '.join(exp_options[i] for i in selected_ids)}", color_by=all_labels)
else:
    render_scatter_3d(coords, all_labels, all_texts, title=f"{method.upper()} 3D — {', '.join(exp_options[i] for i in selected_ids)}", color_by=all_labels)
