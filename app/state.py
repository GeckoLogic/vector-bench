"""Typed session state helpers with pipeline stage management."""
from enum import IntEnum
import streamlit as st
import numpy as np
from vectorbench.experiment.config import ExperimentConfig


class PipelineStage(IntEnum):
    EMPTY = 0
    DATASET = 1
    CHUNKED = 2
    EMBEDDED = 3
    STORED = 4


def _init_defaults():
    defaults = {
        "pipeline_stage": PipelineStage.EMPTY,
        "dataset_name": None,
        "dataset_text": None,
        "chunks": None,
        "config": None,
        "embeddings": None,
        "store": None,
        "models": {},
        "saved_experiments": {},
        "reduction_cache": {},
        "_registry_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def restore_from_registry():
    """On first load per session, restore persisted experiments from disk registry + ChromaDB."""
    _init_defaults()
    if st.session_state._registry_loaded:
        return
    st.session_state._registry_loaded = True

    from vectorbench.experiment.registry import list_experiments
    from vectorbench.store.chroma import ChromaVectorStore

    entries = list_experiments()
    for entry in entries:
        exp_id = entry.get("id")
        if not exp_id or exp_id in st.session_state.saved_experiments:
            continue
        try:
            config = ExperimentConfig.from_dict(entry)
            store = ChromaVectorStore(
                collection_name=config.collection_name,
                persist_directory="data/chroma_db",
            )
            if store.count() == 0:
                continue
            all_data = store.get_all()
            pairs = sorted(
                zip(all_data["metadatas"], all_data["documents"], all_data["embeddings"]),
                key=lambda x: x[0].get("chunk_index", x[0].get("_idx", 0)),
            )
            chunks = [p[1] for p in pairs]
            embeddings = np.array([p[2] for p in pairs], dtype=np.float32)
            st.session_state.saved_experiments[exp_id] = {
                "config": config,
                "chunks": chunks,
                "embeddings": embeddings,
                "store": store,
            }
        except Exception:
            pass  # Skip corrupted or missing collections silently


def advance_to(stage: PipelineStage, **kwargs):
    """Set pipeline stage and update state; clears downstream stages."""
    _init_defaults()
    if stage <= PipelineStage.DATASET:
        st.session_state.chunks = None
        st.session_state.embeddings = None
        st.session_state.store = None
        st.session_state.pipeline_stage = PipelineStage.EMPTY
    if stage <= PipelineStage.CHUNKED:
        st.session_state.embeddings = None
        st.session_state.store = None
    for key, val in kwargs.items():
        st.session_state[key] = val
    st.session_state.pipeline_stage = stage


def get_pipeline_stage() -> PipelineStage:
    _init_defaults()
    return st.session_state.pipeline_stage


def get_dataset_text() -> str | None:
    _init_defaults()
    return st.session_state.dataset_text


def get_chunks() -> list[str] | None:
    _init_defaults()
    return st.session_state.chunks


def get_embeddings() -> np.ndarray | None:
    _init_defaults()
    return st.session_state.embeddings


def get_store():
    _init_defaults()
    return st.session_state.store


def get_config() -> ExperimentConfig | None:
    _init_defaults()
    return st.session_state.config


def get_current_experiment() -> dict | None:
    _init_defaults()
    config = st.session_state.config
    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings
    store = st.session_state.store
    if config and chunks is not None and embeddings is not None and store is not None:
        return {"config": config, "chunks": chunks, "embeddings": embeddings, "store": store}
    return None


def set_current_experiment(config: ExperimentConfig, chunks: list[str], embeddings: np.ndarray, store):
    _init_defaults()
    st.session_state.config = config
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.store = store
    st.session_state.pipeline_stage = PipelineStage.STORED
    exp_id = config.collection_name
    st.session_state.saved_experiments[exp_id] = {
        "config": config, "chunks": chunks, "embeddings": embeddings, "store": store,
    }


def delete_experiment_from_session(exp_id: str):
    """Remove experiment from session state and disk registry."""
    _init_defaults()
    st.session_state.saved_experiments.pop(exp_id, None)
    cfg = st.session_state.get("config")
    if cfg and cfg.collection_name == exp_id:
        st.session_state.config = None
        st.session_state.chunks = None
        st.session_state.embeddings = None
        st.session_state.store = None
        st.session_state.pipeline_stage = PipelineStage.DATASET
    st.session_state.reduction_cache = {
        k: v for k, v in st.session_state.reduction_cache.items()
        if exp_id not in k[0]
    }
    from vectorbench.experiment.registry import delete_experiment
    delete_experiment(exp_id)
