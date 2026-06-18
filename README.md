# Section B — Retrieval pipeline

Implementation of an end-to-end retrieval pipeline over a collection of textual
Wikipedia-style entries. The system receives a batch of queries and returns, for each
query, a ranked list of relevant page id values.

## Setup

```bash
cd student@dpagpu2025s-0038:~/our_solution/Section_B/Section_B  #WE NEED TO CLEAN THIS AFTERWARDS SO THAT IT DOESNT HAVE SECTION_b DUPLICATE
pip install -r requirements.txt
```
### GPU (CUDA) torch
 
To pull a CUDA-enabled torch build, `requirements.txt` includes a PyTorch index
and a CUDA-tagged pin:
 
```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
```
 
`pip install -r requirements.txt` then installs the GPU wheel automatically. 

Models download from the Hugging Face Hub on first use and are cached after:
`all-MiniLM-L6-v2` (bi-encoder, 384-dim) and `ms-marco-MiniLM-L6-v2`
(cross-encoder).

## Build index (offline)

Run once locally to create `artifacts/`. 

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

```bash
python scripts/eval_public.py
```

This loads `artifacts/`, runs `run()` over `data/public_queries.json`, and reports mean NDCG@10.

## Our Pipeline (what `run()` does at query time)

1. Embed the query batch.
2. Dot-product against all chunk vectors, then average per page.
3. Keep the top `5 × K` candidate pages.
4. Re-rank: cross-encoder scores the top 3 chunks of each candidate, averaged per page.
5. Return the top `K = 10` page IDs per query (best first).
