"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import entry_text, EMBEDDING_MODEL_NAME


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str
    tokens: int


_tokenizer: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer | None:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return _tokenizer


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into retrieval units.

    Default: single chunk per page (no chunking). Override for fixed-size or
    semantic chunking strategies.
    """
    page_id = int(record["page_id"])
    text = entry_text(record)

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)

    configs = [
        (512, 150),
        (256, 50),
        (128, 10)
    ]
    chunks: List[Chunk] = []
    chunk_id = 0
    for window_size, overlap in configs:
        if n <= window_size:
            chunk_text = tokenizer.decode(tokens)
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text, tokens=n))
            chunk_id += 1
            continue

        start = 0
        step = window_size - overlap
        while start + window_size < n:
            end = start + window_size
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text, tokens=len(chunk_tokens)))
            chunk_id += 1

            if end == n:
                break
            start += step

        last_tokens = tokens[n - window_size: n]
        last_text = tokenizer.decode(last_tokens)
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=last_text, tokens=len(last_tokens)))

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    print("Chunking")

    chunks: List[Chunk] = []
    for record in tqdm(records, total=len(records)):
        chunks.extend(chunk_entry(record))
    print(f"Found {len(chunks)} chunks\n")
    return chunks