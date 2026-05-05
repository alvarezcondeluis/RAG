"""Embedding model benchmarking suite.

Compares retrieval accuracy and runtime performance across multiple
embedding models on chunks pulled from the Neo4j knowledge graph.
"""

from simple_rag.evaluation.embeddings.benchmark import EmbeddingBenchmark
from simple_rag.evaluation.embeddings.models_config import EMBEDDING_MODELS, ModelSpec

__all__ = ["EmbeddingBenchmark", "EMBEDDING_MODELS", "ModelSpec"]
