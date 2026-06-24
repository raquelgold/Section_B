"""Embedding utilities (sentence-transformers/all-MiniLM-L6-v2 only)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME

# Sub-folder (inside artifacts/) where the embedding model is cached locally.
EMBEDDER_SUBDIR = "models/embedder"

_model: SentenceTransformer | None = None


def _embedder_path(artifacts_dir: Optional[Path] = None) -> Path:
    return Path(artifacts_dir or ARTIFACTS_DIR) / EMBEDDER_SUBDIR


def get_model(artifacts_dir: Optional[Path] = None) -> SentenceTransformer:
    """
    Load the embedding model.

    Prefers a locally-saved copy under artifacts/models/embedder (written at
    index-build time) so query-time retrieval never has to hit HuggingFace.
    Falls back to the HF hub name if no local copy exists yet.
    """
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_path = _embedder_path(artifacts_dir)
        if local_path.exists():
            print("Loading embedding model from local artifacts:", local_path)
            _model = SentenceTransformer(
                str(local_path), device=device, local_files_only=True
            )
        else:
            print("Loading embedding model from HuggingFace:", EMBEDDING_MODEL_NAME)
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _model


def save_embedding_model(artifacts_dir: Optional[Path] = None) -> None:
    """Persist the embedding model (weights + tokenizer) under artifacts/."""
    model = get_model(artifacts_dir)
    local_path = _embedder_path(artifacts_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    save = getattr(model, "save_pretrained", None) or model.save
    save(str(local_path))
    print("Saved embedding model to:", local_path)


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized embeddings, shape (n, dim)."""
    print("Embedding")

    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embedded {len(texts)} texts\n")
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size)
