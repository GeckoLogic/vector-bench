import json
from pathlib import Path
from datetime import datetime
from .config import ExperimentConfig

REGISTRY_PATH = Path("data/registry.json")


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save_registry(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2))


def save_experiment(config: ExperimentConfig, extra: dict | None = None) -> str:
    """Save experiment config to registry. Returns experiment ID."""
    registry = _load_registry()
    exp_id = config.collection_name
    entry = config.to_dict()
    if extra:
        entry.update(extra)
    registry[exp_id] = entry
    _save_registry(registry)
    return exp_id


def load_experiment(exp_id: str) -> ExperimentConfig | None:
    """Load experiment config by ID."""
    registry = _load_registry()
    if exp_id not in registry:
        return None
    return ExperimentConfig.from_dict(registry[exp_id])


def list_experiments() -> list[dict]:
    """List all saved experiments."""
    registry = _load_registry()
    return [{"id": k, **v} for k, v in registry.items()]


def delete_experiment(exp_id: str) -> bool:
    """Delete experiment from registry. Returns True if found."""
    registry = _load_registry()
    if exp_id not in registry:
        return False
    del registry[exp_id]
    _save_registry(registry)
    return True
