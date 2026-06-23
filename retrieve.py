"""Query-time retrieval: recall-first dense -> cross-encoder rerank.

Built from the ablations, not the picture:
  * the task is multi-gold (up to 12 relevant pages/query) over 27k pages, so the
    bottleneck is RECALL -- gold pages sit deep in the dense ranking (recall@50
    ~0.46, recall@500 ~0.78). A 50-page rerank pool was starving the reranker.
  * mean-pool dense >> max-pool here (0.38 vs 0.15): max-pool latches onto
    spurious single 128-token chunks across the huge corpus.
  * intro-paragraph clustering is a no-op (27k pages -> ~26.8k singletons), so
    it's gone. So are MaxSim (hurt) and BM25 fusion (hurt as a ranking signal).

Strategy: rank all pages by mean-pooled dense cosine, take a DEEP candidate pool,
optionally union in BM25's top-K for extra recall, then let the cross-encoder
rerank the pool down to the top-k. Knobs at the top.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

# --- knobs ----------------------------------------------------------------
DENSE_POOL = "mean"          # chunk->page aggregation: "mean" (use this) or "max"
CAND_POOL = 50               # dense candidates (union@50 recall ~0.86, fits 60s)
BM25_RECALL_K = 50           # union BM25 top-K -- recovers gold dense misses
CE_CHUNKS_PER_PAGE = 2       # best 2 chunks/page -> 0.4486 (sweep winner, ~40s/50q)
CE_BATCH = 128               # cross-encoder batch size (GPU throughput)
RERANK_MODE = "cross_encoder"  # "cross_encoder" | "none" (dense-only)
# --------------------------------------------------------------------------

_ce_model = None


def _get_ce():
    global _ce_model
    if _ce_model is None:
        import torch
        from sentence_transformers import CrossEncoder
        from utils import CROSS_ENCODER_MODEL_NAME
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device for cross-encoder:", device)
        _ce_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)
    return _ce_model


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    loaded = load_index(artifacts_dir)
    corpus_vectors, corpus_texts, chunk_page_ids = loaded[0], loaded[1], loaded[2]
    query_vectors = embed_queries(queries)

    print("Querying")
    if query_vectors.size == 0:
        return [[] for _ in queries]

    # ---- precompute page universe + chunk->page map (once) ----
    page_order: List[int] = []
    page_chunks: Dict[int, List[int]] = defaultdict(list)
    seen = set()
    for ci, pid in enumerate(chunk_page_ids):
        ip = int(pid)
        if ip not in seen:
            seen.add(ip); page_order.append(ip)
        page_chunks[ip].append(ci)
    num_pages = len(page_order)
    page_idx = {pid: i for i, pid in enumerate(page_order)}
    chunk_to_page = np.array([page_idx[int(p)] for p in chunk_page_ids], dtype=np.int64)

    # ---- optional BM25 recall source ----
    bm25 = None
    bm25_gather = None
    if BM25_RECALL_K > 0:
        from bm25 import load_bm25
        bm25 = load_bm25(artifacts_dir if artifacts_dir is not None else Path("artifacts"))
        bm25_pos = {pid: i for i, pid in enumerate(bm25.page_ids)}
        bm25_gather = np.array([bm25_pos.get(pid, -1) for pid in page_order], dtype=np.int64)

    dense_all = query_vectors @ corpus_vectors.T          # (Q, num_chunks)
    ce = _get_ce() if RERANK_MODE == "cross_encoder" else None

    ranked: List[List[int]] = []
    for qi, q in enumerate(queries):
        row = dense_all[qi]

        # dense page score (mean-pool, or max)
        if DENSE_POOL == "max":
            dense_page = np.full(num_pages, -np.inf, dtype=np.float32)
            np.maximum.at(dense_page, chunk_to_page, row)
        else:
            s = np.zeros(num_pages, dtype=np.float32)
            c = np.zeros(num_pages, dtype=np.float32)
            np.add.at(s, chunk_to_page, row)
            np.add.at(c, chunk_to_page, 1.0)
            dense_page = s / np.maximum(c, 1.0)

        # deep candidate pool by dense rank
        dorder = np.argsort(-dense_page)
        cand = list(dorder[:CAND_POOL])

        # optionally widen recall with BM25's top-K
        if bm25 is not None:
            braw = bm25.score(q)
            bpage = np.where(bm25_gather >= 0, braw[bm25_gather], -np.inf)
            btop = np.argsort(-bpage)[:BM25_RECALL_K]
            seen_c = set(cand)
            for p in btop:
                if p not in seen_c and bpage[p] > 0:
                    cand.append(int(p)); seen_c.add(int(p))

        if ce is None:
            ranked.append([page_order[p] for p in cand[:top_k]])
            continue

        # cross-encoder rerank: best chunk(s) per candidate page
        pairs, owner = [], []
        for p in cand:
            pid = page_order[p]
            cidx = page_chunks[pid]
            best = sorted(cidx, key=lambda c: row[c], reverse=True)[:CE_CHUNKS_PER_PAGE]
            for c in best:
                pairs.append((q, corpus_texts[c]))
                owner.append(p)
        scores = ce.predict(pairs, batch_size=CE_BATCH, show_progress_bar=False)
        agg: Dict[int, list] = defaultdict(list)
        for sc, p in zip(scores, owner):
            agg[p].append(sc)
        ordered = sorted(cand, key=lambda p: np.mean(agg[p]), reverse=True)
        ranked.append([page_order[p] for p in ordered[:top_k]])

    print(f"\nFound {sum(len(r) for r in ranked)} ranked pages\n")
    return ranked