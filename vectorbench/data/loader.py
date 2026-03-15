from pathlib import Path

SAMPLES_DIR = Path(__file__).parent / "samples"


def load_sample_dataset(name: str) -> str:
    """Load a sample dataset by name (without .txt extension)."""
    path = SAMPLES_DIR / f"{name}.txt"
    if not path.exists():
        available = [p.stem for p in SAMPLES_DIR.glob("*.txt")]
        raise FileNotFoundError(
            f"Sample '{name}' not found. Available: {available}"
        )
    return path.read_text(encoding="utf-8")


def list_sample_datasets() -> list[str]:
    """Return list of available sample dataset names."""
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.txt"))


def load_uploaded_text(content: str | bytes) -> str:
    """Normalize uploaded text content to string."""
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content
