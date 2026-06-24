# Section B: 

We created a retrieval pipeline designed for page ranking.

---

### `chunk.py`

This module handles text preprocessing by partitioning raw, variable-length Wikipedia pages into uniform retrieval units.

### `index.py`

This offline module builds the unified storage structures required for rapid query matching. 

### `retrieve.py`

This runtime retrieval engine executes a multi-stage search strategy by scoring and merging results from the dense and lexical tracks.

## 💾 Precomputed Index Artifacts

All essential indexing data structures are stored locally within the repository under the `artifacts/` folder.

Pretrained model weights for `sentence-transformers/all-MiniLM-L6-v2` are fetched dynamically from the HuggingFace hub at runtime.

### Artifact Registry

| Relative Path | Format | Description |
| --- | --- | --- |
| `artifacts/index_vectors.npy` | Binary NumPy (`.npy`) | L2-normalized dense embedding vectors ($\mathbb{R}^{N \times 384}$) generated across all text chunks. |
| `artifacts/index_meta.json` | JSON | Structural metadata containing sequential mappings of `page_ids`, `chunk_ids`, and the source model configurations. |
| `artifacts/lexical_index.npz` | Compressed Archive (`.npz`) | Compact representation containing vocabulary lookups, Inverse Document Frequencies (IDF), token length priors, and an inverted index structured in **Compressed Sparse Row (CSR)** layout (`indptr`, `docs`, `tfs`). |

---

## 🚀 Setup & Execution Guide

### 1. Cloning the Repository

Since the precomputed indexing artifacts are tracked using Git LFS, ensure LFS is initialized on your system before cloning:

```bash
# Ensure Git LFS is installed locally
git lfs install

# Clone the repository
git clone https://github.com/raquelgold/Section_B.git
cd Section_B

```

### 2. Evaluation Execution

Run the evaluation suite directly:

```bash
python scripts/eval_public.py

```

---

## 👥 Collaboration Log

This repository was built by Itav Dan and Raquel Goldsztejn, for more information on our logic process look at our video describing what we did and why:
https://drive.google.com/file/d/1-_BOqB0DFcowDAAsIBAw9TMG25wa3n61/view?usp=sharing
