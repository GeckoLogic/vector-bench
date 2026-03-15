import numpy as np
import pytest
from vectorbench.experiment.config import ExperimentConfig
from vectorbench.experiment.runner import ExperimentRunner
from tests.conftest import MockEmbedder


@pytest.fixture
def config():
    return ExperimentConfig(
        model_name="mock-embedder",
        strategy="fixed",
        chunk_size=20,
        dataset_name="test",
    )


def test_runner_full_pipeline(config, sample_text, mock_embedder):
    runner = ExperimentRunner(
        config=config,
        embedder=mock_embedder,
        in_memory=True,
    )
    result = runner.run(sample_text)
    assert "chunks" in result
    assert "embeddings" in result
    assert "store" in result
    assert result["n_chunks"] > 0
    assert result["embeddings"].shape[0] == result["n_chunks"]


def test_runner_stores_embeddings(config, sample_text, mock_embedder):
    runner = ExperimentRunner(config=config, embedder=mock_embedder, in_memory=True)
    result = runner.run(sample_text)
    store = result["store"]
    assert store.count() == result["n_chunks"]


def test_runner_search(config, sample_text, mock_embedder):
    runner = ExperimentRunner(config=config, embedder=mock_embedder, in_memory=True)
    result = runner.run(sample_text)
    store = result["store"]
    q_emb = mock_embedder.embed(["test query"])[0]
    results = store.query(q_emb, n_results=3)
    assert len(results) == 3


def test_runner_different_strategies(sample_text, mock_embedder):
    for strategy in ["fixed", "overlapping", "sentence", "paragraph", "document"]:
        config = ExperimentConfig(strategy=strategy, dataset_name="test")
        runner = ExperimentRunner(config=config, embedder=mock_embedder, in_memory=True)
        result = runner.run(sample_text)
        assert result["n_chunks"] > 0


def test_runner_progress_callback(config, sample_text, mock_embedder):
    calls = []
    def cb(done, total):
        calls.append((done, total))

    runner = ExperimentRunner(config=config, embedder=mock_embedder, in_memory=True, progress_callback=cb)
    runner.run(sample_text)
    assert len(calls) > 0
