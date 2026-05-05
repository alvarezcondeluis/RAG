"""Catalog of embedding models to benchmark.

Add or comment-out entries here to control which models the benchmark runs.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelSpec:
    hf_id: str
    family: str
    expected_dim: int
    max_seq_length: int = 512
    trust_remote_code: bool = False
    query_prefix: Optional[str] = None
    passage_prefix: Optional[str] = None
    notes: str = ""


EMBEDDING_MODELS: List[ModelSpec] = [
    ModelSpec(
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        family="nomic",
        expected_dim=768,
        max_seq_length=8192,
        trust_remote_code=True,
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        notes="Project default; long context, instruction-tuned. Optimized for SEC domains.",
    ),
    ModelSpec(
        hf_id="nlpaueb/sec-bert-base",
        family="sec-bert",
        expected_dim=768,
        max_seq_length=512,
        notes="BERT pre-trained on SEC EDGAR filings; raw encoder, mean-pooled. Domain-specific.",
    ),
    ModelSpec(
        hf_id="BAAI/bge-base-en-v1.5",
        family="bge",
        expected_dim=768,
        max_seq_length=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Strong general-purpose retrieval model. Balanced accuracy/speed.",
    ),
    ModelSpec(
        hf_id="BAAI/bge-small-en-v1.5",
        family="bge",
        expected_dim=384,
        max_seq_length=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Smaller/faster BGE variant. Good speed/accuracy trade-off for large indexes.",
    ),
    ModelSpec(
        hf_id="BAAI/bge-large-en-v1.5",
        family="bge",
        expected_dim=1024,
        max_seq_length=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Largest BGE variant. Best accuracy, highest memory/latency.",
    ),
    ModelSpec(
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        family="minilm",
        expected_dim=384,
        max_seq_length=256,
        notes="Lightweight baseline. Useful for speed comparisons.",
    ),
]
