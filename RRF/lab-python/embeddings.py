"""Text embeddings with all-MiniLM-L6-v2 (ONNX, CPU), cached on disk."""

from functools import cache
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("/cache")


@cache
def _model():
    return TextEmbedding(MODEL_NAME, cache_dir=str(CACHE_DIR))


def embed(texts, cache_name=None):
    """Embed a list of texts into a (len(texts), 384) array.

    With `cache_name`, the vectors are saved to disk and reused on the next run.
    """
    cache_file = CACHE_DIR / "vectors" / f"{cache_name}.npy" if cache_name else None
    if cache_file and cache_file.exists():
        return np.load(cache_file)

    vectors = np.array(list(_model().embed(texts, batch_size=64)), dtype=np.float32)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vectors)
    return vectors
