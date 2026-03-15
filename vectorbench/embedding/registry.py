MODEL_REGISTRY: dict[str, dict] = {
    "all-MiniLM-L6-v2": {
        "display_name": "MiniLM-L6 (fast, 384d)",
        "dimension": 384,
        "description": "Fast and lightweight. Great for general-purpose semantic similarity.",
        "size_mb": 90,
    },
    "all-mpnet-base-v2": {
        "display_name": "MPNet-base (accurate, 768d)",
        "dimension": 768,
        "description": "Higher quality embeddings, larger model. Best for precision tasks.",
        "size_mb": 420,
    },
    "paraphrase-MiniLM-L3-v2": {
        "display_name": "MiniLM-L3 (tiny, 384d)",
        "dimension": 384,
        "description": "Smallest model. Fast inference, lower quality.",
        "size_mb": 61,
    },
    "all-distilroberta-v1": {
        "display_name": "DistilRoBERTa (balanced, 768d)",
        "dimension": 768,
        "description": "Balanced quality/speed. Good for diverse text types.",
        "size_mb": 290,
    },
}
