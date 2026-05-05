"""Retrieval-quality metrics for embedding benchmarks.

Inputs are dense matrices (numpy arrays) of corpus and query embeddings,
plus a parallel list of gold chunk ids per query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class AccuracyMetrics:
    hit_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    pos_sim_mean: float
    pos_sim_std: float
    neg_sim_mean: float
    neg_sim_std: float
    pos_sim_p10: float
    neg_sim_p90: float
    margin: float
    per_category_recall_at_k: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hit_at_k": {str(k): v for k, v in self.hit_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "mrr": self.mrr,
            "similarity": {
                "positive_mean": self.pos_sim_mean,
                "positive_std": self.pos_sim_std,
                "negative_mean": self.neg_sim_mean,
                "negative_std": self.neg_sim_std,
                "positive_p10": self.pos_sim_p10,
                "negative_p90": self.neg_sim_p90,
                "margin_mean": self.margin,
            },
            "per_category_recall_at_k": {
                cat: {str(k): v for k, v in d.items()}
                for cat, d in self.per_category_recall_at_k.items()
            },
        }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def compute_accuracy_metrics(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    corpus_ids: Sequence[str],
    gold_ids: Sequence[str],
    categories: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> AccuracyMetrics:
    """Compute retrieval metrics from query/corpus embeddings.

    All k_values must be <= len(corpus_ids). The corpus is treated as a
    single shared pool: every query is ranked against all corpus chunks,
    its gold chunk is the single positive, and everything else negative.
    """
    if len(query_embeddings) != len(gold_ids):
        raise ValueError("query_embeddings and gold_ids length mismatch")
    if len(corpus_embeddings) != len(corpus_ids):
        raise ValueError("corpus_embeddings and corpus_ids length mismatch")

    corpus_norm = _l2_normalize(corpus_embeddings.astype(np.float32))
    query_norm = _l2_normalize(query_embeddings.astype(np.float32))

    # cosine similarity since both are L2-normalized
    sim = query_norm @ corpus_norm.T  # (n_queries, n_corpus)

    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    gold_idx = np.array([id_to_idx[g] for g in gold_ids], dtype=np.int64)

    n_queries, n_corpus = sim.shape
    max_k = max(k_values)
    max_k = min(max_k, n_corpus)

    # top-k indices per query
    if max_k >= n_corpus:
        topk_idx = np.argsort(-sim, axis=1)[:, :max_k]
    else:
        part = np.argpartition(-sim, max_k - 1, axis=1)[:, :max_k]
        # sort the partial selection by similarity
        rows = np.arange(n_queries)[:, None]
        topk_idx = part[rows, np.argsort(-sim[rows, part], axis=1)]

    # rank of gold per query (1-indexed); rank > max_k means "not found in top-k"
    matches = topk_idx == gold_idx[:, None]
    found = matches.any(axis=1)
    first_hit = np.where(found, matches.argmax(axis=1) + 1, 0)

    hit_at_k = {}
    recall_at_k = {}
    for k in k_values:
        k_eff = min(k, n_corpus)
        flag = (first_hit > 0) & (first_hit <= k_eff)
        hit_at_k[k] = float(flag.mean()) if n_queries else 0.0
        # recall@k == hit@k when there's exactly one positive per query
        recall_at_k[k] = hit_at_k[k]

    rr = np.where(found, 1.0 / np.maximum(first_hit, 1), 0.0)
    mrr = float(rr.mean()) if n_queries else 0.0

    # similarity distribution
    pos_sim = sim[np.arange(n_queries), gold_idx]
    mask = np.ones_like(sim, dtype=bool)
    mask[np.arange(n_queries), gold_idx] = False
    neg_sim_flat = sim[mask]

    pos_mean = float(pos_sim.mean()) if n_queries else 0.0
    pos_std = float(pos_sim.std()) if n_queries else 0.0
    neg_mean = float(neg_sim_flat.mean()) if neg_sim_flat.size else 0.0
    neg_std = float(neg_sim_flat.std()) if neg_sim_flat.size else 0.0
    pos_p10 = float(np.percentile(pos_sim, 10)) if n_queries else 0.0
    neg_p90 = float(np.percentile(neg_sim_flat, 90)) if neg_sim_flat.size else 0.0

    # per-category breakdown
    per_cat: Dict[str, Dict[int, float]] = {}
    cats = np.array(categories)
    for cat in np.unique(cats):
        idx = np.where(cats == cat)[0]
        if idx.size == 0:
            continue
        cat_first_hit = first_hit[idx]
        per_cat[str(cat)] = {}
        for k in k_values:
            k_eff = min(k, n_corpus)
            flag = (cat_first_hit > 0) & (cat_first_hit <= k_eff)
            per_cat[str(cat)][k] = float(flag.mean())

    return AccuracyMetrics(
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        pos_sim_mean=pos_mean,
        pos_sim_std=pos_std,
        neg_sim_mean=neg_mean,
        neg_sim_std=neg_std,
        pos_sim_p10=pos_p10,
        neg_sim_p90=neg_p90,
        margin=pos_mean - neg_mean,
        per_category_recall_at_k=per_cat,
    )
