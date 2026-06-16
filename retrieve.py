"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder
import torch

from chunk import chunk_entry
from embed import embed_queries
from index import load_index
from utils import K_EVAL, CROSS_ENCODER_MODEL_NAME, iter_entries, entry_text

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    """Lazy initialization of the Cross-Encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device for cross-encoder:", device)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)
    return _cross_encoder


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Default: brute-force dot product on L2-normalized vectors.
    Replace with FAISS / reranking as needed.
    """
    corpus_vectors, corpus_texts, page_ids = load_index(artifacts_dir)
    query_vectors = embed_queries(queries)

    print("Querying")
    if query_vectors.size == 0:
        return [[] for _ in queries]

    top_n = 5 * top_k

    records = list(iter_entries())
    pid_to_text = {int(r["page_id"]): entry_text(r) for r in records}

    scores = query_vectors @ corpus_vectors.T

    unique_pids = []
    seen_pids = set()
    for pid in page_ids:
        if pid not in seen_pids:
            int_pid = int(pid)
            seen_pids.add(int_pid)
            unique_pids.append(int_pid)

    num_unique_pids = len(unique_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    chunk_to_unique_idx = np.array([pid_to_idx[pid] for pid in page_ids], dtype=np.int32)
    chunk_counts = np.bincount(chunk_to_unique_idx, minlength=num_unique_pids)
    safe_counts = np.maximum(chunk_counts, 1)

    cross_encoder = get_cross_encoder()
    ranked: List[List[int]] = []
    for query_idx, row in enumerate(scores):
        query_str = queries[query_idx]

        page_sums = np.zeros(num_unique_pids, dtype=np.float32)
        np.add.at(page_sums, chunk_to_unique_idx, row)
        page_means = page_sums / safe_counts
        top_n_indices = np.argsort(-page_means)[:top_n]
        candidate_pids = [unique_pids[idx] for idx in top_n_indices]

        pairs = []
        for pid in candidate_pids:
            doc_text = pid_to_text.get(int(pid), "")
            pairs.append((query_str, doc_text))

        rerank_scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)

        final_top_indices = np.argsort(-rerank_scores)[:top_k]
        final_pids = [candidate_pids[idx] for idx in final_top_indices]
        ranked.append(final_pids)

    print(f"Found {sum([len(r) for r in ranked])} ranked pages\n")
    return ranked
