#!/usr/bin/env python3
"""CLI entry point for the embedding model benchmark.

Examples
--------
# Default run (all configured models, 30 chunks per Neo4j label, 1 question each)
uv run python -m simple_rag.evaluation.embeddings.run_benchmark

# Quick smoke test (1 model, small corpus)
uv run python -m simple_rag.evaluation.embeddings.run_benchmark \
    --models nomic-ai/nomic-embed-text-v1.5 --per-label 5 --max-query-chunks 10

# Force-regenerate the synthetic test set
uv run python -m simple_rag.evaluation.embeddings.run_benchmark --regenerate-test-set
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

# Make sure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simple_rag.evaluation.embeddings.benchmark import EmbeddingBenchmark
from simple_rag.evaluation.embeddings.models_config import (
    EMBEDDING_MODELS,
    ModelSpec,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark embedding models for the SEC RAG system")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="HuggingFace ids to benchmark. Default = all entries in models_config.EMBEDDING_MODELS.",
    )
    p.add_argument("--per-label", type=int, default=30, help="Chunks pulled per Neo4j label.")
    p.add_argument(
        "--questions-per-chunk", type=int, default=1, help="Synthetic queries per source chunk."
    )
    p.add_argument(
        "--max-query-chunks",
        type=int,
        default=None,
        help="Cap chunks used to generate queries (None = all corpus chunks).",
    )
    p.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10], help="k values for Hit@k.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latency-samples", type=int, default=30)
    p.add_argument(
        "--corpus-labels",
        nargs="*",
        default=None,
        help="Restrict corpus to specific Neo4j labels (e.g. RiskFactor Objective).",
    )
    p.add_argument("--test-set-name", default="default", help="Cache name for the synthetic test set.")
    p.add_argument(
        "--regenerate-test-set",
        action="store_true",
        help="Ignore any cached synthetic test set and rebuild it.",
    )
    return p.parse_args()


def filter_models(model_ids: list[str] | None) -> list[ModelSpec]:
    if not model_ids:
        return list(EMBEDDING_MODELS)
    by_id = {m.hf_id: m for m in EMBEDDING_MODELS}
    out = []
    for mid in model_ids:
        if mid in by_id:
            out.append(by_id[mid])
        else:
            # Allow ad-hoc models not in the config file
            out.append(ModelSpec(hf_id=mid, family="custom", expected_dim=0))
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    models = filter_models(args.models)

    print(f"Models to benchmark ({len(models)}):")
    for m in models:
        print(f"  - {m.hf_id}  [{m.family}]")

    bench = EmbeddingBenchmark(
        models=models,
        per_label=args.per_label,
        questions_per_chunk=args.questions_per_chunk,
        max_query_chunks=args.max_query_chunks,
        k_values=args.k,
        batch_size=args.batch_size,
        latency_samples=args.latency_samples,
        corpus_labels=args.corpus_labels,
        test_set_name=args.test_set_name,
        regenerate_test_set=args.regenerate_test_set,
    )
    bench.run()


if __name__ == "__main__":
    main()
