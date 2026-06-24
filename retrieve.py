from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL, ARTIFACTS_DIR

# --- Globals & Singletons ---
_LOADED_CORPUS_EMBEDDINGS: Optional[np.ndarray] = None
_LOADED_PAGE_IDENTIFIERS: Optional[np.ndarray] = None
_GLOBAL_CLUSTER_INDEX: Optional[AggregatedClusterIndex] = None
_CHUNK_INDEX_TO_CLUSTER_ID_MAP: Optional[np.ndarray] = None


class AggregatedClusterIndex:
    """Manages lexical information and document mappings across document entity clusters."""

    def __init__(self, storage_directory: Optional[Path] = None):
        root_path = Path(storage_directory) if storage_directory is not None else Path(ARTIFACTS_DIR)
        index_archive = np.load(root_path / "lexical_index.npz")

        # Core statistics and inverted index components
        self.cluster_token_lengths = index_archive["dls"].astype(np.float64)
        self.inverse_document_frequencies = index_archive["idf"]
        self.compressed_sparse_row_pointers = index_archive["indptr"]
        self.posting_cluster_ids = index_archive["docs"]
        self.posting_term_frequencies = index_archive["tfs"]

        # Structural metadata payload
        metadata_payload = json.loads(str(index_archive["metadata"]))
        self.cluster_to_page_mappings: List[List[int]] = metadata_payload["clusters"]
        self.vocabulary_lookup: Dict[str, int] = metadata_payload["vocab"]
        self.truncated_cluster_texts: List[str] = metadata_payload["texts"]

        # Calculated helpers
        self.total_clusters_count: int = len(self.cluster_to_page_mappings)
        self.average_cluster_length: float = (
            float(self.cluster_token_lengths.mean()) if self.total_clusters_count else 0.0
        )
        self.logarithm_length_prior = np.log(self.cluster_token_lengths + 1.0)

        # Reverse lookup mapping page IDs to cluster indexes
        self.page_to_cluster_map: Dict[int, int] = {
            int(page_id): cluster_idx
            for cluster_idx, member_pages in enumerate(self.cluster_to_page_mappings)
            for page_id in member_pages
        }

    def compute_bm25_scores(
        self, query_string: str, term_saturation_k1: float = 0.8, length_normalization_b: float = 0.75
    ) -> np.ndarray:
        """Calculates exact BM25 scores for all clusters against the input query."""
        all_cluster_scores = np.zeros(self.total_clusters_count, dtype=np.float64)
        length_penalty_denominator = term_saturation_k1 * (
            1.0 - length_normalization_b + length_normalization_b * self.cluster_token_lengths / self.average_cluster_length
        )
        
        token_pattern = re.compile(r"\b\w+\b")
        unique_query_tokens = set(token_pattern.findall(query_string.lower()))

        for individual_token in unique_query_tokens:
            target_term_id = self.vocabulary_lookup.get(individual_token)
            if target_term_id is None:
                continue
            
            # Slice postings array boundaries for the given term
            pointer_start = int(self.compressed_sparse_row_pointers[target_term_id])
            pointer_end = int(self.compressed_sparse_row_pointers[target_term_id + 1])
            
            matching_cluster_ids = self.posting_cluster_ids[pointer_start:pointer_end]
            term_frequencies = self.posting_term_frequencies[pointer_start:pointer_end].astype(np.float64)
            
            # Vectorized scoring calculation across clusters containing the term
            all_cluster_scores[matching_cluster_ids] += (
                self.inverse_document_frequencies[target_term_id]
                * term_frequencies
                * (term_saturation_k1 + 1.0)
                / (term_frequencies + length_penalty_denominator[matching_cluster_ids])
            )
            
        return all_cluster_scores


def initialize_or_retrieve_indices(artifacts_dir: Optional[Path]) -> tuple:
    """Ensures index assets are cached in memory and returns reference variables."""
    global _LOADED_CORPUS_EMBEDDINGS, _LOADED_PAGE_IDENTIFIERS, _GLOBAL_CLUSTER_INDEX, _CHUNK_INDEX_TO_CLUSTER_ID_MAP
    
    if _LOADED_CORPUS_EMBEDDINGS is None:
        _LOADED_CORPUS_EMBEDDINGS, _LOADED_PAGE_IDENTIFIERS = load_index(artifacts_dir)
        _GLOBAL_CLUSTER_INDEX = AggregatedClusterIndex(artifacts_dir)
        _CHUNK_INDEX_TO_CLUSTER_ID_MAP = np.array(
            [_GLOBAL_CLUSTER_INDEX.page_to_cluster_map.get(int(pid), -1) for pid in _LOADED_PAGE_IDENTIFIERS]
        )
        
    return _LOADED_CORPUS_EMBEDDINGS, _LOADED_PAGE_IDENTIFIERS, _GLOBAL_CLUSTER_INDEX, _CHUNK_INDEX_TO_CLUSTER_ID_MAP


def extract_highest_dense_similarities(
    chunk_scores: np.ndarray, page_ids: np.ndarray, chunk_to_cluster_map: np.ndarray, max_chunks_to_scan: int = 4000
) -> tuple[Dict[int, float], Dict[int, float]]:
    """Identifies peak similarity values per cluster and per document page from vector similarities."""
    partition_pivot = min(max_chunks_to_scan, len(chunk_scores) - 1)
    highest_scoring_indices = np.argpartition(-chunk_scores, partition_pivot)[:max_chunks_to_scan]
    
    peak_cluster_scores: Dict[int, float] = {}
    peak_page_scores: Dict[int, float] = {}
    
    for matching_chunk_idx in highest_scoring_indices:
        similarity_score = float(chunk_scores[matching_chunk_idx])
        assigned_cluster_id = int(chunk_to_cluster_map[int(matching_chunk_idx)])
        
        if assigned_cluster_id >= 0:
            peak_cluster_scores[assigned_cluster_id] = max(similarity_score, peak_cluster_scores.get(assigned_cluster_id, -1e30))
            
        assigned_page_id = int(page_ids[int(matching_chunk_idx)])
        peak_page_scores[assigned_page_id] = max(similarity_score, peak_page_scores.get(assigned_page_id, -1e30))
        
    return peak_cluster_scores, peak_page_scores


def convert_scores_to_rankings_map(score_dictionary: Dict[int, float], maximum_elements: int) -> Dict[int, int]:
    """Generates an item-to-rank map containing up to maximum_elements items."""
    sorted_items = sorted(score_dictionary, key=lambda identifier: -score_dictionary[identifier])[:maximum_elements]
    return {identifier: rank_position for rank_position, identifier in enumerate(sorted_items)}


def evaluate_token_max_similarity(query_token_vectors: np.ndarray, document_token_vectors: np.ndarray) -> float:
    """Calculates late-interaction MaxSim metrics between normalized token matrix segments."""
    normalized_query_tokens = query_token_vectors / (np.linalg.norm(query_token_vectors, axis=1, keepdims=True) + 1e-9)
    normalized_doc_tokens = document_token_vectors / (np.linalg.norm(document_token_vectors, axis=1, keepdims=True) + 1e-9)
    return float((normalized_query_tokens @ normalized_doc_tokens.T).max(axis=1).mean())


def execute_token_level_reranking(
    raw_query: str, Top_ranked_clusters: List[int], current_fusion_scores: Dict[int, float], cluster_ctx: AggregatedClusterIndex
) -> List[int]:
    """Adjusts the top-tier cluster sequence via fine-grained token alignment embeddings."""
    max_rerank_depth = 30
    interaction_blend_beta = 0.3
    
    candidate_head = Top_ranked_clusters[:max_rerank_depth]
    if len(candidate_head) < 2 or not cluster_ctx.truncated_cluster_texts:
        return Top_ranked_clusters
        
    from embed import get_model
    embedding_model = get_model()
    
    query_tokens = embedding_model.encode([raw_query], output_value="token_embeddings")[0]
    document_tokens_list = embedding_model.encode(
        [cluster_ctx.truncated_cluster_texts[cid] for cid in candidate_head], output_value="token_embeddings"
    )
    
    query_tokens_numpy = query_tokens.detach().cpu().numpy()
    maxsim_metrics = np.array([
        evaluate_token_max_similarity(query_tokens_numpy, single_doc_tokens.detach().cpu().numpy())
        for single_doc_tokens in document_tokens_list
    ])
    
    baseline_fusion_scores = np.array([current_fusion_scores[cid] for cid in candidate_head])
    
    # Standardize scale variants
    maxsim_metrics = (maxsim_metrics - maxsim_metrics.mean()) / (maxsim_metrics.std() + 1e-9)
    baseline_fusion_scores = (baseline_fusion_scores - baseline_fusion_scores.mean()) / (baseline_fusion_scores.std() + 1e-9)
    
    combined_scores = -(baseline_fusion_scores + interaction_blend_beta * maxsim_metrics)
    sorted_candidate_head = [candidate_head[index] for index in np.argsort(combined_scores)]
    
    return sorted_candidate_head + Top_ranked_clusters[max_rerank_depth:]


def detect_high_precision_verbatim_terms(query_string: str) -> List[str]:
    """Finds exact alphanumeric expressions within query strings."""
    big_number_expression = re.compile(r"\b\d[\d,]{3,}\b")
    proper_noun_expression = re.compile(r"(?<!^)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
    return big_number_expression.findall(query_string) + proper_noun_expression.findall(query_string)


def apply_reciprocal_rank_fusion(
    lexical_rankings: Dict[int, int],
    semantic_rankings: Dict[int, int],
    cluster_ctx: AggregatedClusterIndex,
    lexical_weight_alpha: float = 0.85,
    rrf_smoothing_constant: int = 60,
    short_length_penalty: float = 0.005,
) -> Dict[int, float]:
    """Combines dual structural rankings into unified scores penalized by length weights."""
    fused_score_registry: Dict[int, float] = {}
    combined_cluster_ids = set(lexical_rankings) | set(semantic_rankings)
    
    for cluster_id in combined_cluster_ids:
        fusion_accumulator = 0.0
        if cluster_id in lexical_rankings:
            fusion_accumulator += lexical_weight_alpha / (rrf_smoothing_constant + lexical_rankings[cluster_id])
        if cluster_id in semantic_rankings:
            fusion_accumulator += (1.0 - lexical_weight_alpha) / (rrf_smoothing_constant + semantic_rankings[cluster_id])
            
        fusion_accumulator -= short_length_penalty * cluster_ctx.logarithm_length_prior[cluster_id]
        fused_score_registry[cluster_id] = fusion_accumulator
        
    return fused_score_registry


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Accepts multiple queries and delivers a list of relevant page IDs per query string."""
    corpus_embeds, page_ids, cluster_index, chunk_to_cluster = initialize_or_retrieve_indices(artifacts_dir)
    query_embeddings = embed_queries(queries)
    
    if query_embeddings.size == 0:
        return [[] for _ in queries]

    similarity_matrix = query_embeddings @ corpus_embeds.T
    aggregated_batch_results: List[List[int]] = []

    for query_index, literal_query in enumerate(queries):
        dense_cluster_max, dense_page_max = extract_highest_dense_similarities(
            similarity_matrix[query_index], page_ids, chunk_to_cluster, max_chunks_to_scan=4000
        )
        
        lexical_cluster_scores = cluster_index.compute_bm25_scores(literal_query)
        lexical_candidate_bound = min(80, len(lexical_cluster_scores) - 1)
        lexical_top_indices = np.argpartition(-lexical_cluster_scores, lexical_candidate_bound)[:80]
        
        filtered_lexical_scores = {
            int(cid): float(lexical_cluster_scores[cid]) for cid in lexical_top_indices if lexical_cluster_scores[cid] > 0
        }

        lexical_rank_map = convert_scores_to_rankings_map(filtered_lexical_scores, maximum_elements=80)
        semantic_rank_map = convert_scores_to_rankings_map(dense_cluster_max, maximum_elements=80)

        fused_scores = apply_reciprocal_rank_fusion(
            lexical_rank_map, semantic_rank_map, cluster_index, lexical_weight_alpha=0.85, rrf_smoothing_constant=60, short_length_penalty=0.005
        )

        # Apply specific bonuses if high-precision tokens appear in content
        matched_exact_tokens = detect_high_precision_verbatim_terms(literal_query)
        if matched_exact_tokens and cluster_index.truncated_cluster_texts:
            for cluster_id in fused_scores:
                target_text = cluster_index.truncated_cluster_texts[cluster_id]
                if any(token_element in target_text for token_element in matched_exact_tokens):
                    fused_scores[cluster_id] += 0.05

        prioritized_clusters = sorted(fused_scores, key=lambda cid: -fused_scores[cid])
        prioritized_clusters = execute_token_level_reranking(literal_query, prioritized_clusters, fused_scores, cluster_index)

        # Assemble and flatten the final target sequence of pages
        query_output_page_ids: List[int] = []
        for cluster_id in prioritized_clusters:
            cluster_member_pages = cluster_index.cluster_to_page_mappings[cluster_id]
            sorted_member_pages = sorted(cluster_member_pages, key=lambda pid: -dense_page_max.get(pid, -1e30))
            
            query_output_page_ids.extend(sorted_member_pages)
            if len(query_output_page_ids) >= top_k:
                break
                
        aggregated_batch_results.append(query_output_page_ids[:top_k])

    return aggregated_batch_results
