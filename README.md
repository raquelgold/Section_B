# Section B — Retrieval pipeline

Two-stage page retrieval over the Wikipedia corpus: a bi-encoder retrieves
candidate pages, a cross-encoder re-ranks them. Scored by **mean NDCG@10** via
`run(queries)` in `main.py`. Index building is offline (not timed); query
embedding + retrieval is the timed path.

## Setup

```bash
cd path/to/student
pip install -r requirements.txt
```

Corpus lives at **`data/Wikipedia Entries/`** (included in the handout).
Models download from the Hugging Face Hub on first use and are cached after:
`all-MiniLM-L6-v2` (bi-encoder, 384-dim) and `ms-marco-MiniLM-L6-v2`
(cross-encoder). GPU is auto-detected; CPU works as a fallback.

## Build index (offline, not timed — your machine only)

Run once locally to create `artifacts/`. **Submit these files** in your repo;
staff do not rebuild the index at grading time.

```bash
python scripts/build_index.py
```

### Artifacts produced (submit all of these)

| Path | Contents |
|------|----------|
| `artifacts/index_vectors.npy` | `(num_chunks, 384)` L2-normalized chunk embeddings. |
| `artifacts/index_meta.json` | Parallel chunk metadata: `page_ids`, `chunk_ids`, `texts`, `model`, `num_vectors`. |

Row `i` of `index_vectors.npy` corresponds to entry `i` in each list in
`index_meta.json`. At query time `retrieve.py` loads both, scores chunks, then
aggregates chunk scores up to `page_id`. Rebuild whenever the corpus or chunking
changes.

## Public self-test

After building, verify a fresh run loads your submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

This loads `artifacts/`, runs `run()` over `data/public_queries.json`, and
reports mean NDCG@10 (cutoff and final list length are `K_EVAL = 10`). A clean
self-test confirms the submitted artifacts are sufficient on their own.

## Pipeline (what `run()` does at query time)

1. Embed the query batch (bi-encoder, L2-normalized).
2. Dot-product against all chunk vectors (= cosine), then average per page.
3. Keep the top `5 × K` candidate pages.
4. Re-rank: cross-encoder scores the top 3 chunks of each candidate, averaged per page.
5. Return the top `K = 10` page IDs per query (best first).

Tunable knobs: `top_n = 5 * top_k` and `top_chunks_per_page = 3` in
`retrieve.py`; model names and `K_EVAL` in `utils.py`.

## Submit

Public GitHub repo with this code, **required** `artifacts/`, and a concise
README documenting artifact paths. See the assignment PDF for video and grading
details.
