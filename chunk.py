"""Preprocessing and chunking of corpus pages into retrieval units."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import EMBEDDING_MODEL_NAME


CHUNK_SIZE = 180
CHUNK_OVERLAP = 25


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


_tokenizer: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer:
    """Lazy initialization of the official model tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return _tokenizer


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into stable, token-bounded sliding windows."""
    page_id = int(record["page_id"])
    title = record.get("title", "").strip()
    content = record.get("content", "").strip()

    if not content:
        return []

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(content, add_special_tokens=False)
    n = len(tokens)

    if n <= CHUNK_SIZE:
        return [Chunk(page_id=page_id, chunk_id=0, text=f"{title}: {content}")]

    chunks: List[Chunk] = []
    chunk_id = 0
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens).strip()

        chunks.append(Chunk(
            page_id=page_id,
            chunk_id=chunk_id,
            text=f"{title}: {chunk_text}"
        ))
        chunk_id += 1

        if end == n:
            break
        start += step

    return chunks

def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in tqdm(records):
        chunks.extend(chunk_entry(record))
    return chunks
