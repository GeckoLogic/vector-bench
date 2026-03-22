"""Page 3: Embedding model + strategy selection, single run, batch run, and export."""
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from vectorbenchapp.state import (
    get_dataset_text, _init_defaults,
    set_current_experiment,
)
from vectorbench.embedding.sentence_transformer import SentenceTransformerEmbedder
from vectorbench.embedding.registry import MODEL_REGISTRY
from vectorbench.experiment.config import ExperimentConfig
from vectorbench.experiment.runner import ExperimentRunner
from vectorbench.experiment.registry import save_experiment
from vectorbenchapp.components.embedding_table import render_embedding_table

_init_defaults()
st.set_page_config(page_title="Embed — VectorBench", layout="wide")
st.title("3. Embed")
st.markdown(
    "Embedding converts each text chunk into a high-dimensional numerical vector that captures its semantic meaning. "
    "Chunks with similar meanings end up with vectors that point in similar directions — which is what makes "
    "semantic search possible. Choose a model and chunking strategy, then run the embedding. "
    "Use **Single Run** to explore one configuration, **Batch Run** to compare multiple models and strategies at once, "
    "or **Export** to download your chunks and vectors for use outside VectorBench."
)

text = get_dataset_text()
if not text:
    st.warning("No dataset loaded. Go to **Dataset** page first.")
    st.stop()

dataset_name = st.session_state.get("dataset_name", "custom")
model_names = list(MODEL_REGISTRY.keys())
STRATEGIES = ["fixed", "overlapping", "sentence", "paragraph", "document"]
_STRATEGY_LABELS = {
    "fixed": "Fixed Size (tokens)",
    "overlapping": "Overlapping (sliding window)",
    "sentence": "Sentence Boundary",
    "paragraph": "Paragraph",
    "document": "Whole Document",
}


def _exp_name(model: str, strategy: str, chunk_size: int, overlap_pct: int,
              sentences: int, para_min: int, dataset: str) -> str:
    suffix = {
        "fixed": f"-{chunk_size}w",
        "overlapping": f"-{chunk_size}w-{overlap_pct}pct",
        "sentence": f"-{sentences}sent",
        "paragraph": f"-{para_min}minch",
        "document": "",
    }[strategy]
    return f"{model}/{strategy}{suffix}/{dataset}"


def _load_model(model_name: str) -> SentenceTransformerEmbedder:
    if model_name not in st.session_state.models:
        with st.spinner(f"Loading {MODEL_REGISTRY[model_name]['display_name']}..."):
            st.session_state.models[model_name] = SentenceTransformerEmbedder(model_name)
    return st.session_state.models[model_name]


def _run_single(config: ExperimentConfig, embedder, text: str, progress_bar=None, status=None) -> dict:
    def update_progress(done, total):
        if progress_bar:
            progress_bar.progress(done / total)
        if status:
            status.caption(f"Embedding {done}/{total} chunks...")

    runner = ExperimentRunner(
        config=config,
        embedder=embedder,
        progress_callback=update_progress,
    )
    return runner.run(text)


tab_single, tab_batch, tab_export = st.tabs(["Single Run", "Batch Run", "Export"])

# ── Single Run ──────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("Configure one model + strategy combination and run it.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chunking")
        strategy = st.selectbox(
            "Strategy", STRATEGIES,
            format_func=lambda x: _STRATEGY_LABELS[x],
            key="s_strategy",
        )
        chunk_size = 100
        chunk_overlap = 0
        overlap_pct = 0
        sentences_per_chunk = 3
        paragraph_min_length = 50
        if strategy == "fixed":
            chunk_size = st.slider("Chunk Size (words)", 20, 500, 100, key="s_cs")
        elif strategy == "overlapping":
            chunk_size = st.slider("Chunk Size (words)", 20, 500, 100, key="s_cs2")
            overlap_pct = st.slider("Overlap %", 0, 50, 20, key="s_ov")
            chunk_overlap = int(chunk_size * overlap_pct / 100)
            st.caption(f"Overlap: {chunk_overlap} words")
        elif strategy == "sentence":
            sentences_per_chunk = st.slider("Sentences per Chunk", 1, 10, 3, key="s_sp")
        elif strategy == "paragraph":
            paragraph_min_length = st.slider("Min Paragraph Length (chars)", 10, 200, 50, key="s_ml")
        elif strategy == "document":
            st.info("Whole-document: text is treated as a single chunk.")

    with col2:
        st.subheader("Model")
        model_name = st.selectbox(
            "Embedding Model", model_names,
            format_func=lambda x: MODEL_REGISTRY[x]["display_name"],
            key="s_model",
        )
        info = MODEL_REGISTRY[model_name]
        st.caption(info["description"])
        st.caption(f"Dimension: {info['dimension']}d | ~{info['size_mb']}MB")
        _recommended = _exp_name(
            model_name, strategy, chunk_size, overlap_pct,
            sentences_per_chunk, paragraph_min_length, dataset_name,
        )

        # Auto-update the name when params change, unless the user has typed
        # a custom name (i.e. the current value has diverged from the previous
        # recommended name).
        _prev_rec = st.session_state.get("s_expname_recommended")
        _cur = st.session_state.get("s_expname")
        if _cur is None or _cur == _prev_rec:
            st.session_state["s_expname"] = _recommended
        st.session_state["s_expname_recommended"] = _recommended

        exp_name = st.text_input("Experiment Name", key="s_expname")

        if st.session_state.get("s_expname") != _recommended:
            st.button(
                "↩ Use recommended name",
                key="s_expname_reset",
                on_click=lambda: st.session_state.update(
                    {"s_expname": st.session_state.get("s_expname_recommended", "")}
                ),
            )

    if st.button("Run Embedding", type="primary", key="s_run"):
        config = ExperimentConfig(
            model_name=model_name,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sentences_per_chunk=sentences_per_chunk,
            paragraph_min_length=paragraph_min_length,
            dataset_name=dataset_name,
            experiment_name=exp_name,
        )
        embedder = _load_model(model_name)
        progress_bar = st.progress(0)
        status = st.empty()
        start = time.time()
        result = _run_single(config, embedder, text, progress_bar, status)
        elapsed = time.time() - start
        progress_bar.progress(1.0)
        status.success(f"Done in {elapsed:.1f}s | {result['n_chunks']} chunks | {result['embedding_dim']}d")
        set_current_experiment(config, result["chunks"], result["embeddings"], result["store"])
        save_experiment(config)

    from vectorbenchapp.state import get_chunks, get_embeddings
    chunks = get_chunks()
    embeddings = get_embeddings()
    if chunks and embeddings is not None:
        st.markdown("---")
        st.subheader("Embedding Preview")
        render_embedding_table(chunks, embeddings)

# ── Batch Run ────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("Select multiple models × strategies to run all combinations.")

    b_models = st.multiselect(
        "Models",
        model_names,
        default=[model_names[0]],
        format_func=lambda x: MODEL_REGISTRY[x]["display_name"],
        key="b_models",
    )
    b_strategies = st.multiselect(
        "Strategies", STRATEGIES,
        default=["fixed", "sentence"],
        format_func=lambda x: _STRATEGY_LABELS[x],
        key="b_strategies",
    )

    b_chunk_size = st.slider("Chunk Size (words) — Fixed, Overlapping", 20, 500, 100, key="b_cs")
    b_overlap_pct = st.slider("Overlap % — Overlapping", 0, 50, 20, key="b_ov")
    b_sentences = st.slider("Sentences per Chunk — Sentence Boundary", 1, 10, 3, key="b_sp")
    b_paragraph_min_length = st.slider("Min Paragraph Length (chars) — Paragraph", 10, 200, 50, key="b_ml")

    combos = [(m, s) for m in b_models for s in b_strategies]
    if combos:
        st.caption(f"{len(combos)} combination(s) selected")

    if combos and st.button("Run Batch", type="primary", key="b_run"):
        results_table = []
        overall = st.progress(0)
        grid_placeholder = st.empty()

        for i, (m, s) in enumerate(combos):
            overlap = int(b_chunk_size * b_overlap_pct / 100)
            config = ExperimentConfig(
                model_name=m,
                strategy=s,
                chunk_size=b_chunk_size,
                chunk_overlap=overlap,
                sentences_per_chunk=b_sentences,
                paragraph_min_length=b_paragraph_min_length,
                dataset_name=dataset_name,
                experiment_name=_exp_name(
                    m, s, b_chunk_size, b_overlap_pct,
                    b_sentences, b_paragraph_min_length, dataset_name,
                ),
            )
            row = {"Model": MODEL_REGISTRY[m]["display_name"], "Strategy": s, "Status": "Running..."}
            results_table.append(row)
            grid_placeholder.dataframe(pd.DataFrame(results_table), width="stretch")

            try:
                embedder = _load_model(m)
                start = time.time()
                result = _run_single(config, embedder, text)
                elapsed = time.time() - start
                set_current_experiment(config, result["chunks"], result["embeddings"], result["store"])
                save_experiment(config)
                row.update({
                    "Status": "Done",
                    "Chunks": result["n_chunks"],
                    "Dim": result["embedding_dim"],
                    "Time (s)": round(elapsed, 1),
                })
            except Exception as e:
                row.update({"Status": f"Error: {e}"})

            overall.progress((i + 1) / len(combos))
            grid_placeholder.dataframe(pd.DataFrame(results_table), width="stretch")

        overall.progress(1.0)
        st.success(f"Batch complete — {len(combos)} experiments.")

# ── Export ───────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("Download chunks + embeddings or experiment config for any saved experiment.")

    saved = st.session_state.get("saved_experiments", {})
    if not saved:
        st.info("No experiments to export yet. Run an embedding first.")
        st.stop()

    exp_ids = list(saved.keys())
    exp_options = {eid: saved[eid]["config"].experiment_name for eid in exp_ids}
    export_id = st.selectbox(
        "Select experiment", exp_ids,
        format_func=lambda x: exp_options[x],
        key="export_sel",
    )

    exp = saved[export_id]
    cfg: ExperimentConfig = exp["config"]
    chunks: list[str] = exp["chunks"]
    embeddings: np.ndarray = exp["embeddings"]

    st.caption(
        f"{cfg.model_name} | {cfg.strategy} | {len(chunks)} chunks | "
        f"{embeddings.shape[1]}d | created {cfg.created_at[:19]}"
    )

    col_csv, col_json = st.columns(2)

    with col_csv:
        df = pd.DataFrame({"chunk_index": range(len(chunks)), "text": chunks})
        emb_df = pd.DataFrame(
            embeddings,
            columns=[f"emb_{i}" for i in range(embeddings.shape[1])],
        )
        csv_df = pd.concat([df, emb_df], axis=1)
        csv_bytes = csv_df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV (chunks + embeddings)",
            data=csv_bytes,
            file_name=f"{export_id}_embeddings.csv",
            mime="text/csv",
        )

    with col_json:
        json_bytes = cfg.to_json().encode()
        st.download_button(
            "Download Config JSON",
            data=json_bytes,
            file_name=f"{export_id}_config.json",
            mime="application/json",
        )
