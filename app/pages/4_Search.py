"""Page 4: Semantic search against the vector store."""
import streamlit as st
import pandas as pd
from app.state import _init_defaults
from vectorbench.embedding.sentence_transformer import SentenceTransformerEmbedder
from vectorbench.embedding.registry import MODEL_REGISTRY

_init_defaults()
st.set_page_config(page_title="Search — VectorBench", layout="wide")
st.title("4. Search")
st.markdown(
    "Semantic search works by embedding your query into the same vector space as your chunks, "
    "then finding the chunks whose vectors are closest to the query vector. "
    "Unlike keyword search, it finds *meaning* matches rather than exact word matches — "
    "so a query about 'machine cognition' can surface chunks about 'artificial intelligence' "
    "even if those exact words don't appear in the query. "
    "The similarity score tells you how closely each result matches the query's meaning."
)

saved = st.session_state.get("saved_experiments", {})
if not saved:
    st.warning("No embeddings available. Run the **Embed** step first.")
    st.stop()

exp_ids = list(saved.keys())
exp_options = {eid: saved[eid]["config"].experiment_name for eid in exp_ids}

selected_id = st.selectbox(
    "Experiment",
    exp_ids,
    format_func=lambda x: exp_options[x],
)

exp = saved[selected_id]
config = exp["config"]
store = exp["store"]

st.caption(f"Model: **{config.model_name}** | Strategy: **{config.strategy}**")

query = st.text_input("Enter search query", placeholder="What is artificial intelligence?")
n_results = st.slider("Number of results", 1, 20, 5)
metric = st.radio("Similarity metric", ["cosine", "dot_product"], horizontal=True)
_metric_help = {
    "cosine": (
        "**Cosine similarity** measures the *angle* between two vectors, ignoring their length. "
        "A score of 1.0 means the vectors point in exactly the same direction (identical meaning); "
        "0.0 means they are orthogonal (unrelated). "
        "Because it ignores magnitude, cosine similarity is robust to chunks of different lengths "
        "and is the standard choice for most semantic search tasks."
    ),
    "dot_product": (
        "**Dot product** multiplies corresponding dimensions of two vectors and sums the results. "
        "Unlike cosine similarity, it is sensitive to vector *magnitude* — longer vectors (which often "
        "correspond to longer or more information-dense chunks) will score higher regardless of direction. "
        "This can be useful when chunk length is a meaningful signal, but it can also skew results toward "
        "longer chunks. Only use dot product if your embeddings are normalised (magnitude = 1), "
        "in which case it produces the same ranking as cosine."
    ),
}
st.caption(_metric_help[metric])

if query and st.button("Search", type="primary"):
    if config.model_name not in st.session_state.get("models", {}):
        with st.spinner("Loading model..."):
            if "models" not in st.session_state:
                st.session_state.models = {}
            st.session_state.models[config.model_name] = SentenceTransformerEmbedder(config.model_name)
    embedder = st.session_state.models[config.model_name]
    q_emb = embedder.embed([query])[0]

    results = store.query(q_emb, n_results=n_results)

    st.markdown(f"### Top {len(results)} Results")
    for i, r in enumerate(results):
        score = r["score"]
        with st.expander(f"#{i+1} — Score: {score:.4f}", expanded=i < 3):
            st.markdown(r["document"])
            meta = r.get("metadata", {})
            if meta:
                st.caption(f"Chunk index: {meta.get('chunk_index', '?')} | Strategy: {meta.get('strategy', '?')}")
