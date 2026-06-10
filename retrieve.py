"""High-performance GPU-accelerated retrieval pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import faiss

from embed import embed_queries
from index import load_index
from utils import K_EVAL

# Global in-memory resources to avoid disk/GPU transfer overhead between steps
_GLOBAL_INDEX: Optional[faiss.Index] = None
_GLOBAL_PAGE_IDS: Optional[np.ndarray] = None

def get_cached_resources(artifacts_dir: Optional[Path] = None) -> Tuple[faiss.Index, np.ndarray]:
    global _GLOBAL_INDEX, _GLOBAL_PAGE_IDS
    if _GLOBAL_INDEX is None:
        cpu_index, _GLOBAL_PAGE_IDS = load_index(artifacts_dir)
        
        # Move FAISS index to GPU if PyTorch confirms CUDA visibility
        if torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                # Clone index structure directly onto Tesla M60 VRAM (Device ID 0)
                _GLOBAL_INDEX = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                print("[INFO] FAISS successfully cloned to Tesla M60 GPU.")
            except Exception as e:
                print(f"[WARNING] Failed to initialize FAISS GPU resources: {e}. Falling back to CPU.")
                _GLOBAL_INDEX = cpu_index
        else:
            _GLOBAL_INDEX = cpu_index
            
    return _GLOBAL_INDEX, _GLOBAL_PAGE_IDS

def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    # 1. Access pre-cached GPU index structures instantly
    faiss_index, page_ids = get_cached_resources(artifacts_dir)
    
    # 2. Extract parallelized query embeddings via CUDA Bi-Encoder
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    # 3. Fast hardware similarity calculation 
    # Retrieving 50 candidates guarantees strong cross-document coverage
    k_retrieve = min(50, faiss_index.ntotal)
    scores_mat, indices_mat = faiss_index.search(query_vectors, k_retrieve)

    ranked: List[List[int]] = []

    # 4. Pure Vectorized Mapping and Exponential Document Accumulation
    for scores, indices in zip(scores_mat, indices_mat):
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]
        valid_scores = scores[valid_mask]

        # Extract page mappings using vectorized NumPy slicing
        matching_pids = page_ids[valid_indices]

        # Score Aggregation: Exponential Weighting
        # Gives higher priority to top semantic matches across fragments
        boosted_scores = np.exp(valid_scores * 12)

        # Vectorized Group-by Sum Accumulation
        unique_pids, inverse_indices = np.unique(matching_pids, return_inverse=True)
        aggregated_scores = np.zeros_like(unique_pids, dtype=np.float32)
        np.add.at(aggregated_scores, inverse_indices, boosted_scores)

        # Extract Top Performing Document References
        top_sorted_indices = np.argsort(aggregated_scores)[::-1][:top_k]
        ranked.append(unique_pids[top_sorted_indices].tolist())

    return ranked