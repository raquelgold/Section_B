"""Offline index build and load with dynamic device adaptability."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple
import faiss
import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ensure_artifacts_dir, iter_entries

FAISS_INDEX_NAME = "faiss.index"
INDEX_META_NAME = "index_meta.json"

def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int], List[int], List[str]]:
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    
    #vectors = embed_texts(texts)
    # Pass show_progress=True only during offline building!
    vectors = embed_texts(texts, batch_size=512, show_progress=True)
    page_ids = [c.page_id for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    dim = vectors.shape[1]
    # Standard IndexFlatIP provides precise inner-product for L2 normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(out_dir / FAISS_INDEX_NAME))

    meta = {
        "page_ids": page_ids,
        "chunk_ids": chunk_ids,
        "num_vectors": len(vectors),
    }
    
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta), encoding="utf-8")
    return vectors, page_ids, chunk_ids, texts

def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, np.ndarray]:
    """Load index mapping details directly into structured memory spaces."""
    root = artifacts_dir or ARTIFACTS_DIR
    index = faiss.read_index(str(root / FAISS_INDEX_NAME))
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = np.array(meta["page_ids"], dtype=np.int32)
    return index, page_ids