"""Page 5: Metrics dashboard and side-by-side comparison."""
import streamlit as st
import pandas as pd
import plotly.express as px
from app.state import _init_defaults
from vectorbench.embedding.sentence_transformer import SentenceTransformerEmbedder
from vectorbench.evaluation.comparison import compare_experiments, summary_stats

_init_defaults()
st.set_page_config(page_title="Evaluate — VectorBench", layout="wide")
st.title("5. Evaluate")
st.markdown(
    "Different chunking strategies and embedding models can produce very different retrieval results — "
    "even on the same text and queries. This page lets you run the same queries against two saved experiments "
    "side by side so you can directly compare which configuration surfaces more relevant chunks. "
    "The summary statistics show structural differences (chunk count, average similarity) while the "
    "query comparison table reveals how each experiment responds to real questions. "
    "Use this to understand the trade-offs between strategies before settling on one."
)

saved = st.session_state.get("saved_experiments", {})
if len(saved) < 2:
    st.warning("Run at least two experiments in the **Embed** step to compare them here.")
    st.stop()

exp_ids = list(saved.keys())
exp_options = {eid: saved[eid]["config"].experiment_name for eid in exp_ids}

col1, col2 = st.columns(2)
with col1:
    id_a = st.selectbox("Experiment A", exp_ids, format_func=lambda x: exp_options[x])
with col2:
    b_ids = [eid for eid in exp_ids if eid != id_a]
    id_b = st.selectbox("Experiment B", b_ids, format_func=lambda x: exp_options[x])

exp_a = saved[id_a]
exp_b = saved[id_b]

_METRIC_LABELS = {
    "n_chunks":          "Chunk Count",
    "mean_chunk_len":    "Avg Chunk Length (chars)",
    "embedding_dim":     "Embedding Dimension",
    "mean_pairwise_sim": "Mean Pairwise Similarity",
}

_METRIC_EXPLANATIONS = {
    "Chunk Count": (
        "The total number of chunks the text was split into. "
        "More chunks means finer-grained retrieval but more vectors to search."
    ),
    "Avg Chunk Length (chars)": (
        "The average number of characters per chunk. "
        "Longer chunks carry more context per vector; shorter chunks are more precise."
    ),
    "Embedding Dimension": (
        "The size of each embedding vector (number of values). "
        "A 384-dimensional vector has 384 numbers representing the chunk's meaning. "
        "Higher dimensions can capture more nuance but use more memory."
    ),
    "Mean Pairwise Similarity": (
        "The average cosine similarity between every pair of chunk embeddings. "
        "A high value (close to 1.0) means most chunks are semantically similar to each other — "
        "the content is topically tight. "
        "A low value means the content is diverse. "
        "When comparing experiments, a lower value often indicates the chunking strategy is "
        "producing more distinct, non-redundant chunks."
    ),
}

# Summary stats
st.subheader("Summary Statistics")
stats_a = summary_stats(exp_a)
stats_b = summary_stats(exp_b)
if stats_a and stats_b:
    df_stats = pd.DataFrame([
        {
            "Metric": _METRIC_LABELS.get(k, k),
            "Experiment A": round(v, 4) if isinstance(v, float) else v,
            "Experiment B": round(stats_b.get(k, 0), 4) if isinstance(stats_b.get(k), float) else stats_b.get(k),
        }
        for k, v in stats_a.items()
    ])
    st.dataframe(df_stats, width="stretch")

    with st.expander("What do these metrics mean?"):
        for label, explanation in _METRIC_EXPLANATIONS.items():
            st.markdown(f"**{label}** — {explanation}")

    # Bar chart comparison
    metrics = ["mean_pairwise_sim", "n_chunks"]
    chart_data = pd.DataFrame({
        "Metric": [_METRIC_LABELS[m] for m in metrics] * 2,
        "Value": [stats_a.get(m, 0) for m in metrics] + [stats_b.get(m, 0) for m in metrics],
        "Experiment": [exp_options[id_a]] * len(metrics) + [exp_options[id_b]] * len(metrics),
    })
    fig = px.bar(chart_data, x="Metric", y="Value", color="Experiment", barmode="group", title="Metric Comparison")
    st.plotly_chart(fig, width="stretch")

# Query comparison
st.subheader("Query Comparison")
st.markdown(
    "Enter one or more questions below and click **Compare** to run each query against both experiments. "
    "Each query is embedded and used to search the vector store — the top matching chunk is returned "
    "for each experiment. Comparing the results side by side reveals how the choice of chunking strategy "
    "and embedding model affects *which* content is surfaced and *how confidently* it is matched. "
    "A higher score means the returned chunk is more semantically similar to the query."
)

queries_input = st.text_area("Queries (one per line)", "What is artificial intelligence?\nHow does machine learning work?\nWhat are neural networks?")
queries = [q.strip() for q in queries_input.strip().split("\n") if q.strip()]

if queries and st.button("Compare", type="primary"):
    cfg_a = exp_a["config"]
    cfg_b = exp_b["config"]

    if cfg_a.model_name not in st.session_state.models:
        st.session_state.models[cfg_a.model_name] = SentenceTransformerEmbedder(cfg_a.model_name)
    if cfg_b.model_name not in st.session_state.models:
        st.session_state.models[cfg_b.model_name] = SentenceTransformerEmbedder(cfg_b.model_name)

    emb_a = st.session_state.models[cfg_a.model_name].embed(queries)
    emb_b = st.session_state.models[cfg_b.model_name].embed(queries)

    df = compare_experiments(exp_a, exp_b, queries, emb_a, emb_b)
    st.session_state["eval_df"] = df
    st.session_state["eval_exp_a_name"] = exp_options[id_a]
    st.session_state["eval_exp_b_name"] = exp_options[id_b]

if "eval_df" in st.session_state:
    df = st.session_state["eval_df"]
    name_a = st.session_state.get("eval_exp_a_name", "Experiment A")
    name_b = st.session_state.get("eval_exp_b_name", "Experiment B")
    show_full = st.toggle("Show full chunk text", value=False, key="eval_full_text")

    if show_full:
        queries = df["query"].tolist()
        top1_a  = df["top1_a"].tolist()
        top1_b  = df["top1_b"].tolist()
    else:
        queries = [q[:60]  + ("..." if len(q)  > 60  else "") for q in df["query"]]
        top1_a  = [t[:120] + ("..." if len(t)  > 120 else "") for t in df["top1_a"]]
        top1_b  = [t[:120] + ("..." if len(t)  > 120 else "") for t in df["top1_b"]]

    # MultiIndex: experiment name spans "Top-1" and "Score" in a top header row.
    # The actual column headers are just two short words, keeping the table narrow.
    columns = pd.MultiIndex.from_tuples([
        (name_a, "Top-1"),
        (name_a, "Score"),
        (name_b, "Top-1"),
        (name_b, "Score"),
    ])
    display_df = pd.DataFrame(
        list(zip(top1_a, df["score_a"].round(4), top1_b, df["score_b"].round(4))),
        columns=columns,
        index=queries,
    )
    display_df.index.name = "Query"

    styler = (
        display_df.style
        .set_properties(subset=[(name_a, "Top-1"), (name_a, "Score")],
                        **{"background-color": "#dbeafe"})
        .set_properties(subset=[(name_b, "Top-1"), (name_b, "Score")],
                        **{"background-color": "#fef3c7"})
    )

    st.markdown(
        '<div style="overflow-x:auto;">'
        + styler.to_html(sparse_columns=True)
        + "</div>",
        unsafe_allow_html=True,
    )
