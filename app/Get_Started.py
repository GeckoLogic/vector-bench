"""VectorBench — Embedding Strategy Experimentation Platform."""
import streamlit as st

st.set_page_config(
    page_title="VectorBench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.state import _init_defaults, restore_from_registry, delete_experiment_from_session

_init_defaults()
restore_from_registry()  # Reload persisted experiments on every fresh session

st.title("🔬 VectorBench")
st.markdown(
    """
    **Embedding Strategy Experimentation Platform**

    Compare how different text chunking strategies and embedding models affect
    semantic similarity, clustering, and retrieval performance.

    ---

    ### Getting Started

    Use the sidebar to navigate through the pipeline:

    1. **Dataset** — Upload or select sample text
    2. **Chunking Playground** — Explore chunking strategies live
    3. **Embed** — Run embedding models on your chunks
    4. **Search** — Query the vector store
    5. **Evaluate** — Compare experiments
    6. **Explorer** — Visualize the embedding space
    """
)

with st.sidebar:
    st.markdown("---")
    st.subheader("Saved Experiments")
    saved = st.session_state.get("saved_experiments", {})
    if saved:
        for exp_id, exp in list(saved.items()):
            cfg = exp.get("config")
            if not cfg:
                continue
            col_text, col_btn = st.columns([5, 1])
            with col_text:
                st.caption(f"**{cfg.experiment_name}**")
                st.caption(f"{cfg.model_name} | {cfg.strategy} | {len(exp.get('chunks', []))} chunks")
            with col_btn:
                if st.button("×", key=f"del_{exp_id}", help="Delete experiment"):
                    delete_experiment_from_session(exp_id)
                    st.rerun()
    else:
        st.caption("No experiments yet.")
