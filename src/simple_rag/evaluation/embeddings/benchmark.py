"""End-to-end embedding model benchmark.

For each configured embedding model we:
  1. Load the model.
  2. Encode the corpus (passages) and the test queries.
  3. Measure encoding throughput, single-query latency, peak memory.
  4. Score retrieval accuracy (Hit@k, MRR, similarity distribution).
  5. Persist a JSON + human-readable report under reports/.
"""

from __future__ import annotations

import gc
import json
import logging
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

import numpy as np

from simple_rag.evaluation.embeddings.corpus_loader import (
    CorpusChunk,
    load_corpus_from_neo4j,
)
from simple_rag.evaluation.embeddings.metrics import (
    AccuracyMetrics,
    compute_accuracy_metrics,
)
from simple_rag.evaluation.embeddings.models_config import (
    EMBEDDING_MODELS,
    ModelSpec,
)
from simple_rag.evaluation.embeddings.query_generator import (
    QAPair,
    generate_test_set,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    encode_throughput_docs_per_sec: float
    encode_total_time_sec: float
    query_latency_ms_mean: float
    query_latency_ms_p50: float
    query_latency_ms_p95: float
    peak_memory_mb: float
    embedding_dim: int
    estimated_index_size_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelResult:
    spec: ModelSpec
    performance: PerformanceMetrics
    accuracy: AccuracyMetrics
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model": {
                "hf_id": self.spec.hf_id,
                "family": self.spec.family,
                "expected_dim": self.spec.expected_dim,
                "max_seq_length": self.spec.max_seq_length,
                "notes": self.spec.notes,
            },
            "performance": self.performance.to_dict() if self.performance else None,
            "accuracy": self.accuracy.to_dict() if self.accuracy else None,
            "error": self.error,
        }


HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"
TEST_SETS_DIR = HERE / "test_sets"


class EmbeddingBenchmark:
    def __init__(
        self,
        models: Optional[Sequence[ModelSpec]] = None,
        per_label: int = 30,
        questions_per_chunk: int = 1,
        max_query_chunks: Optional[int] = None,
        k_values: Sequence[int] = (1, 3, 5, 10),
        batch_size: int = 32,
        latency_samples: int = 30,
        corpus_labels: Optional[List[str]] = None,
        test_set_name: str = "default",
        regenerate_test_set: bool = False,
    ):
        self.models = list(models or EMBEDDING_MODELS)
        self.per_label = per_label
        self.questions_per_chunk = questions_per_chunk
        self.max_query_chunks = max_query_chunks
        self.k_values = tuple(sorted(set(k_values)))
        self.batch_size = batch_size
        self.latency_samples = latency_samples
        self.corpus_labels = corpus_labels
        self.test_set_name = test_set_name
        self.regenerate_test_set = regenerate_test_set

        self.corpus: List[CorpusChunk] = []
        self.test_set: List[QAPair] = []
        self.results: List[ModelResult] = []

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def prepare_data(self) -> None:
        logger.info("Loading corpus from Neo4j (per_label=%d)", self.per_label)
        self.corpus = load_corpus_from_neo4j(
            per_label=self.per_label, labels=self.corpus_labels
        )
        if not self.corpus:
            raise RuntimeError("Corpus is empty — Neo4j returned no chunks.")
        logger.info("Corpus loaded: %d chunks", len(self.corpus))

        cache_path = TEST_SETS_DIR / f"{self.test_set_name}.json"
        if self.regenerate_test_set and cache_path.exists():
            cache_path.unlink()

        logger.info("Building test set (cache: %s)", cache_path)
        self.test_set = generate_test_set(
            corpus=self.corpus,
            questions_per_chunk=self.questions_per_chunk,
            max_chunks=self.max_query_chunks,
            cache_path=cache_path,
        )
        if not self.test_set:
            raise RuntimeError("Test set generation produced 0 pairs.")
        logger.info("Test set ready: %d query/answer pairs", len(self.test_set))

        # Filter the corpus to only chunks that we actually use as gold
        # — keep all of them so we have realistic distractors.
        chunk_id_set = {c.chunk_id for c in self.corpus}
        missing = [p for p in self.test_set if p.relevant_chunk_id not in chunk_id_set]
        if missing:
            logger.warning(
                "%d gold chunks missing from corpus, dropping those pairs", len(missing)
            )
            self.test_set = [p for p in self.test_set if p.relevant_chunk_id in chunk_id_set]

    # ------------------------------------------------------------------ #
    # Per-model evaluation
    # ------------------------------------------------------------------ #
    def evaluate_model(self, spec: ModelSpec) -> ModelResult:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info("=" * 70)
        logger.info("Evaluating: %s", spec.hf_id)
        logger.info("=" * 70)

        # Lazy load
        try:
            tracemalloc.start()
            model = HuggingFaceEmbedding(
                model_name=spec.hf_id,
                trust_remote_code=spec.trust_remote_code,
                cache_folder="./cache",
                max_length=spec.max_seq_length,
            )
        except Exception as e:
            tracemalloc.stop()
            logger.error("Failed to load %s: %s", spec.hf_id, e)
            return ModelResult(
                spec=spec,
                performance=None,  # type: ignore[arg-type]
                accuracy=None,  # type: ignore[arg-type]
                error=f"load_error: {e}",
            )

        try:
            # ---- Encode corpus ---------------------------------------- #
            passages = [
                (spec.passage_prefix or "") + c.text for c in self.corpus
            ]
            queries = [
                (spec.query_prefix or "") + p.query for p in self.test_set
            ]

            t0 = time.time()
            corpus_embs = self._batched_encode(model, passages, self.batch_size)
            encode_time = time.time() - t0

            query_embs = self._batched_encode(model, queries, self.batch_size)

            # ---- Single-query latency --------------------------------- #
            sample = queries[: self.latency_samples] or queries[:1]
            latencies: List[float] = []
            for q in sample:
                t = time.time()
                _ = model.get_query_embedding(q)
                latencies.append((time.time() - t) * 1000)
            latencies_arr = np.array(latencies)

            # ---- Memory ----------------------------------------------- #
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 * 1024)

            # ---- Performance struct ----------------------------------- #
            corpus_arr = np.asarray(corpus_embs, dtype=np.float32)
            query_arr = np.asarray(query_embs, dtype=np.float32)

            dim = corpus_arr.shape[1] if corpus_arr.ndim == 2 else 0
            est_index_mb = (corpus_arr.nbytes) / (1024 * 1024)
            perf = PerformanceMetrics(
                encode_throughput_docs_per_sec=(
                    len(passages) / encode_time if encode_time > 0 else 0.0
                ),
                encode_total_time_sec=encode_time,
                query_latency_ms_mean=float(latencies_arr.mean()),
                query_latency_ms_p50=float(np.percentile(latencies_arr, 50)),
                query_latency_ms_p95=float(np.percentile(latencies_arr, 95)),
                peak_memory_mb=peak_mb,
                embedding_dim=dim,
                estimated_index_size_mb=est_index_mb,
            )

            # ---- Accuracy --------------------------------------------- #
            corpus_ids = [c.chunk_id for c in self.corpus]
            gold_ids = [p.relevant_chunk_id for p in self.test_set]
            categories = [p.category for p in self.test_set]
            acc = compute_accuracy_metrics(
                corpus_embeddings=corpus_arr,
                query_embeddings=query_arr,
                corpus_ids=corpus_ids,
                gold_ids=gold_ids,
                categories=categories,
                k_values=self.k_values,
            )

            return ModelResult(spec=spec, performance=perf, accuracy=acc)

        except Exception as e:
            logger.exception("Evaluation failed for %s", spec.hf_id)
            try:
                tracemalloc.stop()
            except Exception:
                pass
            return ModelResult(
                spec=spec,
                performance=None,  # type: ignore[arg-type]
                accuracy=None,  # type: ignore[arg-type]
                error=f"eval_error: {e}",
            )
        finally:
            del model
            gc.collect()
            try:
                import torch  # noqa: F401
                import torch as _t

                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _batched_encode(model, texts: List[str], batch_size: int) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out.extend(model.get_text_embedding_batch(batch))
        return out

    # ------------------------------------------------------------------ #
    # Run + report
    # ------------------------------------------------------------------ #
    def run(self) -> List[ModelResult]:
        self.prepare_data()
        for spec in self.models:
            result = self.evaluate_model(spec)
            self.results.append(result)
            self._log_result(result)
        self._write_reports()
        return self.results

    @staticmethod
    def _log_result(r: ModelResult) -> None:
        if r.error:
            logger.warning("[FAIL] %s — %s", r.spec.hf_id, r.error)
            return
        logger.info(
            "[OK] %s | dim=%d | hit@5=%.3f | mrr=%.3f | %.1f docs/sec | %.1f MB peak",
            r.spec.hf_id,
            r.performance.embedding_dim,
            r.accuracy.hit_at_k.get(5, 0.0),
            r.accuracy.mrr,
            r.performance.encode_throughput_docs_per_sec,
            r.performance.peak_memory_mb,
        )

    def _write_reports(self) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = REPORTS_DIR / f"embedding_benchmark_{ts}.json"
        txt_path = REPORTS_DIR / f"embedding_benchmark_{ts}.txt"

        json_payload: Dict[str, Any] = {
            "timestamp": ts,
            "config": {
                "per_label": self.per_label,
                "questions_per_chunk": self.questions_per_chunk,
                "max_query_chunks": self.max_query_chunks,
                "k_values": list(self.k_values),
                "batch_size": self.batch_size,
                "test_set_name": self.test_set_name,
                "corpus_size": len(self.corpus),
                "test_set_size": len(self.test_set),
            },
            "results": [r.to_dict() for r in self.results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self._format_text_report(ts))

        logger.info("Reports written:\n  %s\n  %s", json_path, txt_path)

    def _format_text_report(self, ts: str) -> str:
        lines = []
        lines.append("=" * 90)
        lines.append(f"EMBEDDING MODEL BENCHMARK — {ts}")
        lines.append("=" * 90)
        lines.append(f"Corpus size : {len(self.corpus)} chunks")
        lines.append(f"Test queries: {len(self.test_set)}")
        lines.append(f"k values    : {list(self.k_values)}")
        lines.append("")

        # summary table
        header = (
            f"{'Model':45} {'Dim':>5} {'Hit@1':>7} {'Hit@5':>7} "
            f"{'Hit@10':>7} {'MRR':>7} {'docs/s':>8} {'qLat ms':>9} {'Mem MB':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.results:
            if r.error:
                lines.append(f"{r.spec.hf_id:45} ERROR: {r.error}")
                continue
            p, a = r.performance, r.accuracy
            lines.append(
                f"{r.spec.hf_id:45} {p.embedding_dim:>5d} "
                f"{a.hit_at_k.get(1, 0):>7.3f} {a.hit_at_k.get(5, 0):>7.3f} "
                f"{a.hit_at_k.get(10, 0):>7.3f} {a.mrr:>7.3f} "
                f"{p.encode_throughput_docs_per_sec:>8.1f} "
                f"{p.query_latency_ms_mean:>9.1f} {p.peak_memory_mb:>8.1f}"
            )

        lines.append("")
        lines.append("=" * 90)
        lines.append("PER-MODEL DETAIL")
        lines.append("=" * 90)
        for r in self.results:
            lines.append("")
            lines.append(f"# {r.spec.hf_id} ({r.spec.family})")
            if r.spec.notes:
                lines.append(f"  notes: {r.spec.notes}")
            if r.error:
                lines.append(f"  ERROR: {r.error}")
                continue
            p, a = r.performance, r.accuracy
            lines.append(
                f"  perf: dim={p.embedding_dim} | "
                f"throughput={p.encode_throughput_docs_per_sec:.1f} docs/sec | "
                f"latency mean={p.query_latency_ms_mean:.1f} ms p95={p.query_latency_ms_p95:.1f} ms | "
                f"peak mem={p.peak_memory_mb:.1f} MB | "
                f"index≈{p.estimated_index_size_mb:.1f} MB"
            )
            lines.append(
                f"  accuracy: MRR={a.mrr:.3f} | "
                + " | ".join(f"Hit@{k}={a.hit_at_k[k]:.3f}" for k in self.k_values)
            )
            lines.append(
                f"  similarity: pos μ={a.pos_sim_mean:.3f}±{a.pos_sim_std:.3f} | "
                f"neg μ={a.neg_sim_mean:.3f}±{a.neg_sim_std:.3f} | "
                f"margin={a.margin:.3f} | pos p10={a.pos_sim_p10:.3f} | neg p90={a.neg_sim_p90:.3f}"
            )
            if a.per_category_recall_at_k:
                lines.append("  per-category Hit@5:")
                for cat, kmap in sorted(a.per_category_recall_at_k.items()):
                    lines.append(f"    - {cat:25} {kmap.get(5, 0):.3f}")
        return "\n".join(lines) + "\n"
