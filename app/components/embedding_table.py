"""DataFrame display for chunks + embedding preview."""
import streamlit as st
import numpy as np
import pandas as pd


def render_embedding_table(chunks: list[str], embeddings: np.ndarray, n_preview: int = 5):
    """Show chunks with first N embedding dimensions."""
    if not chunks or embeddings is None:
        st.info("No embeddings to display.")
        return

    dim = embeddings.shape[1]
    preview_cols = [f"emb[{i}]" for i in range(min(n_preview, dim))]

    show_full = st.toggle("Show full chunk text", value=False, key="emb_table_full_text")
    st.caption(f"Embedding dimension: {dim}d | Showing first {min(n_preview, dim)} values")

    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "Chunk #": i,
            "Chunk Text": chunk if show_full else chunk[:80] + ("..." if len(chunk) > 80 else ""),
            "Words": len(chunk.split()),
            **{col: round(float(emb[j]), 4) for j, col in enumerate(preview_cols)},
        })
    df = pd.DataFrame(rows)

    if show_full:
        # st.table renders a plain HTML table whose cells wrap text naturally,
        # showing the complete content without any fixed-height clipping.
        st.table(df)
    else:
        st.dataframe(
            df,
            width="stretch",
            column_config={
                "Chunk #": st.column_config.NumberColumn("Chunk #", width="small"),
                "Chunk Text": st.column_config.TextColumn("Chunk Text", width="large"),
                "Words": st.column_config.NumberColumn("Words", width="small"),
            },
        )
