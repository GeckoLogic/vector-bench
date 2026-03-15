"""Page 1: Dataset selection and upload."""
import streamlit as st
from app.state import advance_to, PipelineStage
from vectorbench.data.loader import list_sample_datasets, load_sample_dataset, load_uploaded_text

st.set_page_config(page_title="Dataset — VectorBench", layout="wide")
st.title("1. Dataset")
st.markdown(
    "Every embedding pipeline starts with raw text. This step is where you choose what content "
    "will flow through the rest of the pipeline — from chunking to embedding to search and evaluation. "
    "Select one of the built-in sample datasets or upload your own `.txt` file, preview the content, "
    "then click **Use This Dataset →** to make it available to the downstream steps."
)

source = st.radio("Source", ["Sample Dataset", "Upload Text"], horizontal=True)

text = None
dataset_name = None

if source == "Sample Dataset":
    datasets = list_sample_datasets()
    dataset_name = st.selectbox("Choose dataset", datasets)
    if dataset_name:
        text = load_sample_dataset(dataset_name)
        show_full = st.toggle("Show full text", value=False)
        preview = text if show_full else text[:1000] + ("..." if len(text) > 1000 else "")
        st.text_area("Preview", preview, height=400 if show_full else 200, disabled=True)
        st.caption(f"{len(text)} characters | {len(text.split())} words")
else:
    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded:
        text = load_uploaded_text(uploaded.read())
        dataset_name = uploaded.name.rsplit(".", 1)[0]
        show_full = st.toggle("Show full text", value=False)
        preview = text if show_full else text[:1000] + ("..." if len(text) > 1000 else "")
        st.text_area("Preview", preview, height=400 if show_full else 200, disabled=True)
        st.caption(f"{len(text)} characters | {len(text.split())} words")
    else:
        st.info("Upload a .txt file to continue.")

if text and st.button("Use This Dataset →", type="primary"):
    advance_to(
        PipelineStage.DATASET,
        dataset_name=dataset_name,
        dataset_text=text,
    )
    st.success(f"Dataset '{dataset_name}' loaded. Navigate to Chunking Playground.")
