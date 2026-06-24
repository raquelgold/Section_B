"""
Offline index build pipeline.
Generates dense embeddings and a compressed structural lexical index.
"""
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

# Output artifact filenames
DENSE_VECTORS_FILE = "index_vectors.npy"
DENSE_META_FILE = "index_meta.json"
LEXICAL_INDEX_FILE = "lexical_index.npz"

# Constants for text processing
TITLE_BOOST_FACTOR = 2
MAX_TEXT_CHARS = 1800


def _tokenize(text: str) -> List[str]:
    """Standard alphanumeric tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def _extract_intro_paragraph(content: str) -> str:
    """Isolate the first structural paragraph as the entity signature."""
    parts = content.split("\n\n", 1)
    return parts[0].strip().lower() if parts else ""


from collections import Counter


def _compile_lexical_index(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Group documents into entity clusters and build a unified compressed lexical index archive."""

    # Step 1: Group related pages by their semantic intro paragraph signature
    intro_to_page_ids_map: Dict[str, List[int]] = {}
    for record_item in records:
        current_page_id = int(record_item["page_id"])
        intro_signature = _extract_intro_paragraph(record_item.get("content", ""))
        lookup_key = intro_signature if intro_signature else f"__solo_{current_page_id}"
        intro_to_page_ids_map.setdefault(lookup_key, []).append(current_page_id)

    page_id_to_record_lookup = {int(rec["page_id"]): rec for rec in records}
    grouped_entity_clusters: List[List[int]] = list(intro_to_page_ids_map.values())

    # Step 2: Aggregate text and tokenize across cluster members using list comprehensions
    tokenized_clusters_collection: List[List[str]] = []
    truncated_cluster_texts_history: List[str] = []

    for cluster_members in grouped_entity_clusters:
        # Reconstruct repeated titles and collect contents via clean list comprehensions
        combined_raw_strings = [
            " ".join([member_rec.get("title", "")] * TITLE_BOOST_FACTOR + [member_rec.get("content", "")])
            for member_id in cluster_members
            if (member_rec := page_id_to_record_lookup.get(member_id))
        ]

        # Tokenize the unified text block representing the entire entity cluster
        unified_cluster_string = " ".join(combined_raw_strings)
        tokenized_clusters_collection.append(_tokenize(unified_cluster_string))
        truncated_cluster_texts_history.append(unified_cluster_string[:MAX_TEXT_CHARS])

    # Step 3: Build global vocabulary registry maps and capture tracking statistics
    total_clusters_count = len(grouped_entity_clusters)
    global_vocabulary_registry: Dict[str, int] = {}
    term_document_frequencies: Dict[int, int] = {}
    inverted_index_postings: Dict[int, List[Tuple[int, int]]] = {}

    # Pythonic list comprehension for document lengths array population
    cluster_token_lengths = np.array([len(token_list) for token_list in tokenized_clusters_collection], dtype=np.int64)

    for current_cluster_index, cluster_tokens_list in enumerate(tokenized_clusters_collection):
        # Using Counter to calculate local frequencies instead of nested loops with containment checks
        local_term_counts = Counter(cluster_tokens_list)

        for distinct_token, frequency_count in local_term_counts.items():
            if distinct_token not in global_vocabulary_registry:
                global_vocabulary_registry[distinct_token] = len(global_vocabulary_registry)

            assigned_vocabulary_term_id = global_vocabulary_registry[distinct_token]
            term_document_frequencies[assigned_vocabulary_term_id] = term_document_frequencies.get(
                assigned_vocabulary_term_id, 0) + 1
            inverted_index_postings.setdefault(assigned_vocabulary_term_id, []).append(
                (current_cluster_index, frequency_count))

    # Step 4: Compute Inverse Document Frequencies (IDF) over all discovered terms
    total_vocabulary_size = len(global_vocabulary_registry)
    inverse_document_frequencies = np.zeros(total_vocabulary_size, dtype=np.float32)
    for term_id, doc_freq in term_document_frequencies.items():
        inverse_document_frequencies[term_id] = math.log(
            1.0 + (total_clusters_count - doc_freq + 0.5) / (doc_freq + 0.5))

    # Step 5: Convert postings registry maps to flat array streams (Compressed Sparse Row layout)
    compressed_index_pointers = np.zeros(total_vocabulary_size + 1, dtype=np.int64)
    for term_id in range(total_vocabulary_size):
        compressed_index_pointers[term_id + 1] = compressed_index_pointers[term_id] + len(
            inverted_index_postings.get(term_id, []))

    aggregate_postings_count = int(compressed_index_pointers[-1])
    flat_postings_cluster_indices = np.empty(aggregate_postings_count, dtype=np.int32)
    flat_postings_term_frequencies = np.empty(aggregate_postings_count, dtype=np.int32)

    for term_id in range(total_vocabulary_size):
        base_write_position = int(compressed_index_pointers[term_id])
        current_term_postings = inverted_index_postings.get(term_id, [])

        for write_offset, (associated_cluster_id, local_frequency) in enumerate(current_term_postings):
            target_index = base_write_position + write_offset
            flat_postings_cluster_indices[target_index] = associated_cluster_id
            flat_postings_term_frequencies[target_index] = local_frequency

    # Step 6: Serialize everything securely into a single compressed binary package archive
    np.savez_compressed(
        out_dir / LEXICAL_INDEX_FILE,
        dls=cluster_token_lengths,
        idf=inverse_document_frequencies,
        indptr=compressed_index_pointers,
        docs=flat_postings_cluster_indices,
        tfs=flat_postings_term_frequencies,
        metadata=np.array(json.dumps({
            "clusters": grouped_entity_clusters,
            "vocab": global_vocabulary_registry,
            "texts": truncated_cluster_texts_history
        }))
    )


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Generate dense embeddings and build the consolidated lexical index archive."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))

    # Process dense embeddings chunk index components
    chunks = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    # Save dense artifacts
    np.save(out_dir / DENSE_VECTORS_FILE, vectors)
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / DENSE_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Compile and store unified lexical index stats
    _compile_lexical_index(records, out_dir)
    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed dense vector representations and map arrays."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / DENSE_VECTORS_FILE)
    meta = json.loads((root / DENSE_META_FILE).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids
