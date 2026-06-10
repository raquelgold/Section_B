"""Token-aware sentence window chunking with title context."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import EMBEDDING_MODEL_NAME

CHUNK_SIZE = 180  # Ideal density for MiniLM
CHUNK_OVERLAP = 40

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str

_tokenizer: Optional[AutoTokenizer] = None

def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return _tokenizer

def split_into_sentences(text: str) -> List[str]:
    """Basic sentence splitter using regex boundary markers."""
    sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_end.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    title_prefix = f"Title: {title}. Content: " if title else ""
    
    if not content:
        return [Chunk(page_id=page_id, chunk_id=0, text=title)]

    sentences = split_into_sentences(content)
    tokenizer = _get_tokenizer()
    
    # Pre-tokenize sentences to accurately gauge sizing
    sentence_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in sentences]
    
    chunks: List[Chunk] = []
    chunk_id = 0
    
    current_chunk_tokens: List[int] = []
    current_chunk_words: List[str] = []
    
    # Sliding sentence window compilation
    for i, (sentence, tokens) in enumerate(zip(sentences, sentence_tokens)):
        current_chunk_words.append(sentence)
        current_chunk_tokens.extend(tokens)
        
        if len(current_chunk_tokens) >= CHUNK_SIZE or i == len(sentences) - 1:
            combined_body = " ".join(current_chunk_words)
            full_text = f"{title_prefix}{combined_body}"
            
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=full_text))
            chunk_id += 1
            
            # Slide window back by shifting elements out based on overlap targets
            keep_idx = 0
            temp_tokens = current_chunk_tokens.copy()
            while keep_idx < len(current_chunk_words) - 1:
                dropped_sentence_len = len(sentence_tokens[i - len(current_chunk_words) + 1 + keep_idx])
                if len(temp_tokens) - dropped_sentence_len < CHUNK_OVERLAP:
                    break
                temp_tokens = temp_tokens[dropped_sentence_len:]
                keep_idx += 1
                
            current_chunk_words = current_chunk_words[keep_idx + 1:]
            current_chunk_tokens = temp_tokens

    if not chunks:
        chunks.append(Chunk(page_id=page_id, chunk_id=0, text=f"{title_prefix}{content[:400]}"))
        
    return chunks

def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in tqdm(records, desc="Chunking corpus"):
        chunks.extend(chunk_entry(record))
    return chunks