from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import json


@dataclass
class ExperimentConfig:
    """Central configuration passed through the entire pipeline."""
    model_name: str = "all-MiniLM-L6-v2"
    strategy: str = "fixed"
    chunk_size: int = 100
    chunk_overlap: int = 0
    sentences_per_chunk: int = 3
    paragraph_min_length: int = 50
    similarity_metric: str = "cosine"
    dataset_name: str = "wikipedia_ai"
    experiment_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"{self.model_name}/{self.strategy}/{self.dataset_name}"

    @property
    def config_repr(self) -> str:
        """Deterministic string representation for hashing."""
        parts = [
            self.model_name,
            self.strategy,
            str(self.chunk_size),
            str(self.chunk_overlap),
            str(self.sentences_per_chunk),
            str(self.paragraph_min_length),
            self.dataset_name,
        ]
        return "|".join(parts)

    @property
    def collection_name(self) -> str:
        """Deterministic ChromaDB collection name."""
        h = hashlib.sha256(self.config_repr.encode()).hexdigest()[:12]
        return f"vb_{h}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
