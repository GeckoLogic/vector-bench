"""Page 2: Live chunking strategy comparison."""
import re
import streamlit as st
from app.state import get_dataset_text, get_pipeline_stage, PipelineStage
from app.components.chunk_preview import render_chunk_preview, render_chunk_stats
from vectorbench.chunking import (
    FixedChunker, OverlappingChunker, SentenceChunker, ParagraphChunker, DocumentChunker
)

st.set_page_config(page_title="Chunking Playground — VectorBench", layout="wide")
st.title("2. Chunking Playground")

text = get_dataset_text()
if not text:
    st.warning("No dataset loaded. Go to **Dataset** page first.")
    st.stop()

st.markdown(
    "Before text can be embedded, it must be split into smaller pieces called *chunks*. "
    "The chunking strategy you choose directly affects what each embedding vector represents — "
    "and therefore how well semantic search and retrieval will work. "
    "The Chunking Playground lets you experiment with different strategies and instantly see how your "
    "text gets divided, so you can build an intuition before committing to a strategy on the Embed page."
)

_STRATEGY_INFO = {
    "fixed": {
        "label": "Fixed Size (tokens)",
        "summary": "Splits text into chunks of exactly N words. When the word count reaches the limit, a new chunk begins — regardless of sentence or paragraph boundaries.",
        "details": (
            "**How it works:** The text is whitespace-tokenised into words. "
            "Words are accumulated until the chunk reaches `chunk_size`; any remaining words start the next chunk. "
            "Because splits happen mid-sentence, chunks may begin or end mid-thought.\n\n"
            "**Best for:** Baseline comparisons, simple pipelines, or when you want predictable, uniform-length vectors."
        ),
    },
    "overlapping": {
        "label": "Overlapping (sliding window)",
        "summary": "Like Fixed Size, but each new chunk re-uses the last N words of the previous one, so context is never abruptly cut off at a boundary.",
        "details": (
            "**How it works:** A window of `chunk_size` words advances by `chunk_size − overlap` words each step. "
            "The overlap region appears in two consecutive chunks, bridging the boundary. "
            "Overlap is set as a percentage of chunk size.\n\n"
            "**Best for:** Retrieval tasks where an answer might straddle a chunk boundary. "
            "Higher overlap improves recall at the cost of more chunks and redundant embeddings."
        ),
    },
    "sentence": {
        "label": "Sentence Boundary",
        "summary": "Groups complete sentences together, so no chunk ever splits mid-sentence. Boundaries are detected by NLTK's Punkt tokeniser.",
        "details": (
            "**How it works:** NLTK's `sent_tokenize` (trained Punkt model) detects sentence endings using "
            "punctuation, capitalisation, and abbreviation heuristics. "
            "Every `sentences_per_chunk` complete sentences are joined into one chunk. "
            "Chunks vary in word length but always contain whole sentences.\n\n"
            "**Best for:** Question-answering and summarisation where semantic coherence within a chunk matters more than uniform size."
        ),
    },
    "paragraph": {
        "label": "Paragraph",
        "summary": "Treats each paragraph as a chunk. A paragraph boundary is any sequence of two or more newline characters (i.e. a blank line).",
        "details": (
            "**How it works:** The text is split on `\\n\\n` (or longer runs of whitespace-only lines). "
            "Resulting fragments are stripped; any fragment shorter than `min_length` characters is discarded "
            "(merged into the next paragraph in the output list).\n\n"
            "**Best for:** Documents with natural paragraph structure — articles, essays, documentation — "
            "where each paragraph expresses a distinct idea. Chunk sizes vary widely."
        ),
    },
    "document": {
        "label": "Whole Document",
        "summary": "The entire text becomes a single chunk. No splitting is performed.",
        "details": (
            "**How it works:** The text is returned as-is (with leading/trailing whitespace stripped). "
            "The result is always exactly one chunk regardless of document length.\n\n"
            "**Best for:** Short documents, document-level similarity comparisons, or as a control condition "
            "when you want to measure how splitting affects retrieval quality."
        ),
    },
}

def _md_to_html(text: str) -> str:
    """Convert the subset of markdown used in details strings to HTML."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\n\n", r"</p><p style='margin:6px 0 0 0;'>", text)
    return text

col_controls, col_gap, col_preview = st.columns([1, 0.08, 2])

chunker = None

with col_controls:
    with st.container(border=True):
        strategy = st.selectbox(
            "Chunking Strategy",
            list(_STRATEGY_INFO.keys()),
            format_func=lambda x: _STRATEGY_INFO[x]["label"],
        )

        info = _STRATEGY_INFO[strategy]
        summary_html = re.sub(r"`(.+?)`", r"<code>\1</code>", info["summary"])
        details_html = _md_to_html(info["details"])
        st.markdown(
            f"""
            <div style="background:rgba(28,131,225,0.1);border-left:4px solid rgb(28,131,225);
                        border-radius:4px;padding:12px 16px;margin-bottom:8px;">
              <p style="margin:0 0 8px 0;">ℹ️ {summary_html}</p>
              <details>
                <summary style="cursor:pointer;color:rgb(28,131,225);font-weight:600;
                                user-select:none;list-style:none;">
                  ▶ How this strategy works
                </summary>
                <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(28,131,225,0.2);">
                  <p style="margin:0;">{details_html}</p>
                </div>
              </details>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if strategy == "fixed":
            chunk_size = st.slider("Chunk Size (words)", 20, 500, 100)
            chunker = FixedChunker(chunk_size=chunk_size)

        elif strategy == "overlapping":
            chunk_size = st.slider("Chunk Size (words)", 20, 500, 100)
            overlap_pct = st.slider("Overlap %", 0, 50, 20)
            overlap = int(chunk_size * overlap_pct / 100)
            st.caption(f"Overlap: {overlap} words")
            chunker = OverlappingChunker(chunk_size=chunk_size, overlap=overlap)

        elif strategy == "sentence":
            sentences_per = st.slider("Sentences per Chunk", 1, 10, 3)
            chunker = SentenceChunker(sentences_per_chunk=sentences_per)

        elif strategy == "paragraph":
            min_len = st.slider("Min Paragraph Length (chars)", 10, 200, 50)
            chunker = ParagraphChunker(min_length=min_len)

        elif strategy == "document":
            chunker = DocumentChunker()

with col_preview:
    st.subheader("Chunk Preview")
    if chunker:
        chunks = chunker.chunk(text)
        render_chunk_stats(chunks)
        render_chunk_preview(chunks, original_text=text)
    else:
        st.info("Select a strategy to see the preview.")
