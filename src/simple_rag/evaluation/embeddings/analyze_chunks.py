"""
Chunk quality analysis for the embedding benchmark corpus.

For each category this script reports:
  1. Basic stats  — count, text length distribution
  2. Lexical uniqueness — how much of each chunk is shared vs distinct
  3. Within-category TF-IDF similarity — high mean = hard to distinguish
  4. Sample chunks — shortest / most similar (hardest to retrieve)
  5. Query alignment — from the cached test set, shows query/chunk pairs
     so you can judge whether the question actually matches the text
  6. Improvement suggestions based on the findings

Run from the repo root:
    uv run python simple_rag/evaluation/embeddings/analyze_chunks.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[4] / "src"))

import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from simple_rag.evaluation.embeddings.corpus_loader import (
    CorpusChunk,
    load_corpus_from_neo4j,
)

HERE = Path(__file__).resolve().parent
TEST_SET_PATH = HERE / "test_sets" / "default.json"

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
G  = "\033[92m"   # green
B  = "\033[94m"   # blue
DIM = "\033[2m"
RST = "\033[0m"
BOLD = "\033[1m"

def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    colour = G if value >= 0.6 else (Y if value >= 0.3 else R)
    return colour + "█" * filled + DIM + "░" * (width - filled) + RST

def _colour(value: float, thresholds=(0.3, 0.6)) -> str:
    if value >= thresholds[1]:
        return f"{G}{value:.3f}{RST}"
    if value >= thresholds[0]:
        return f"{Y}{value:.3f}{RST}"
    return f"{R}{value:.3f}{RST}"

def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)

# ── helpers ───────────────────────────────────────────────────────────────────

def _token_set(text: str) -> set:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _uniqueness_scores(texts: List[str]) -> List[float]:
    """Fraction of tokens in each chunk not found in any other chunk."""
    token_sets = [_token_set(t) for t in texts]
    all_tokens: Counter = Counter()
    for ts in token_sets:
        all_tokens.update(ts)
    scores = []
    for ts in token_sets:
        if not ts:
            scores.append(0.0)
            continue
        unique = sum(1 for tok in ts if all_tokens[tok] == 1)
        scores.append(unique / len(ts))
    return scores


def _tfidf_similarity_matrix(texts: List[str]) -> np.ndarray:
    if len(texts) < 2:
        return np.array([[1.0]])
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return np.eye(len(texts))
    return cosine_similarity(X)


def _mean_pairwise_sim(sim_matrix: np.ndarray) -> float:
    n = sim_matrix.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(sim_matrix[mask].mean())


def _top_shared_terms(texts: List[str], n: int = 10) -> List[str]:
    if len(texts) < 2:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=2000)
    try:
        X = vec.fit_transform(texts).toarray()
    except ValueError:
        return []
    feature_names = vec.get_feature_names_out()
    # mean TF-IDF across docs (high mean = term appears with weight in many docs)
    mean_scores = X.mean(axis=0)
    top_idx = mean_scores.argsort()[-n:][::-1]
    return [feature_names[i] for i in top_idx]


# ── per-category analysis ─────────────────────────────────────────────────────

def analyse_category(
    cat: str,
    chunks: List[CorpusChunk],
    qa_pairs: List[Dict],
    verbose: bool = True,
) -> Dict:
    texts = [c.text for c in chunks]
    lengths = [len(t) for t in texts]
    uniqueness = _uniqueness_scores(texts)
    sim_matrix = _tfidf_similarity_matrix(texts)
    mean_sim = _mean_pairwise_sim(sim_matrix)
    shared_terms = _top_shared_terms(texts)

    result = dict(
        category=cat,
        count=len(chunks),
        length_mean=np.mean(lengths),
        length_min=min(lengths),
        length_max=max(lengths),
        length_p25=float(np.percentile(lengths, 25)),
        length_p75=float(np.percentile(lengths, 75)),
        uniqueness_mean=float(np.mean(uniqueness)),
        uniqueness_min=float(np.min(uniqueness)),
        mean_intra_sim=mean_sim,
        shared_terms=shared_terms,
    )

    if not verbose:
        return result

    # ── header ────────────────────────────────────────────────────────────────
    sep = "─" * 90
    print(f"\n{BOLD}{'═'*90}{RST}")
    print(f"{BOLD}  {cat}  ({len(chunks)} chunks){RST}")
    print(sep)

    # ── text length ───────────────────────────────────────────────────────────
    print(f"  {BOLD}Text length{RST}  "
          f"min={lengths[0] if len(lengths)==1 else min(lengths):5d}  "
          f"p25={result['length_p25']:6.0f}  "
          f"mean={result['length_mean']:6.0f}  "
          f"p75={result['length_p75']:6.0f}  "
          f"max={max(lengths):6d}")

    # ── uniqueness ────────────────────────────────────────────────────────────
    uniq_colour = G if result['uniqueness_mean'] > 0.4 else (Y if result['uniqueness_mean'] > 0.2 else R)
    print(f"  {BOLD}Lexical uniqueness{RST}  "
          f"mean={uniq_colour}{result['uniqueness_mean']:.2%}{RST}  "
          f"min={result['uniqueness_min']:.2%}  "
          f"{_bar(result['uniqueness_mean'])}")

    # ── within-category similarity ────────────────────────────────────────────
    # High similarity → hard to distinguish → bad retrieval
    sim_inv = 1 - mean_sim
    sim_colour = G if mean_sim < 0.3 else (Y if mean_sim < 0.5 else R)
    print(f"  {BOLD}Intra-category TF-IDF sim{RST}  "
          f"mean={sim_colour}{mean_sim:.3f}{RST}  "
          f"(lower = easier to retrieve)  "
          f"{_bar(sim_inv)}")

    # ── shared terms ──────────────────────────────────────────────────────────
    if shared_terms:
        print(f"  {BOLD}Top shared terms{RST}  {DIM}{', '.join(shared_terms)}{RST}")

    # ── shortest chunks (most likely to be too generic) ───────────────────────
    sorted_by_len = sorted(zip(lengths, chunks, uniqueness), key=lambda x: x[0])
    print(f"\n  {BOLD}Shortest chunks (highest risk of being too generic):{RST}")
    for ln, chunk, uniq in sorted_by_len[:2]:
        parent_label = f"  [{chunk.parent}]" if chunk.parent else ""
        print(f"    {DIM}len={ln}  uniqueness={uniq:.1%}{parent_label}{RST}")
        print(_wrap(chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""), indent="    "))

    # ── most similar pair ─────────────────────────────────────────────────────
    if len(chunks) >= 2:
        np.fill_diagonal(sim_matrix, 0)
        i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
        max_sim = sim_matrix[i, j]
        print(f"\n  {BOLD}Most similar pair{RST}  sim={R if max_sim > 0.7 else Y}{max_sim:.3f}{RST}")
        for idx, label in [(i, "A"), (j, "B")]:
            parent = f"[{chunks[idx].parent}] " if chunks[idx].parent else ""
            print(f"    {BOLD}{label}{RST} {DIM}{parent}len={len(chunks[idx].text)}{RST}")
            print(_wrap(chunks[idx].text[:250] + "…", indent="    "))

    # ── query / chunk sample ──────────────────────────────────────────────────
    cat_pairs = [p for p in qa_pairs if p["category"] == cat]
    if cat_pairs:
        print(f"\n  {BOLD}Sample query → chunk pairs ({len(cat_pairs)} total):{RST}")
        for pair in cat_pairs[:3]:
            print(f"    {G}Q:{RST} {pair['query']}")
            print(_wrap(pair["source_text"][:200] + "…", indent=f"    {DIM}"))
            print(RST)

    return result


# ── suggestions ───────────────────────────────────────────────────────────────

def print_suggestions(results: List[Dict]) -> None:
    print(f"\n{BOLD}{'═'*90}{RST}")
    print(f"{BOLD}  IMPROVEMENT SUGGESTIONS{RST}")
    print("─" * 90)

    for r in sorted(results, key=lambda x: x["mean_intra_sim"], reverse=True):
        cat = r["category"]
        issues = []
        actions = []

        if r["mean_intra_sim"] > 0.5:
            issues.append(f"very high intra-similarity ({r['mean_intra_sim']:.2f}) → chunks are near-identical")
            actions.append("prepend fund/company name + section title to each chunk text")
            actions.append("consider merging all chunks per fund into one section-level text")

        elif r["mean_intra_sim"] > 0.3:
            issues.append(f"moderate intra-similarity ({r['mean_intra_sim']:.2f}) → hard to distinguish")
            actions.append("add parent entity name as context prefix to chunk text")

        if r["length_mean"] < 300:
            issues.append(f"very short chunks (mean={r['length_mean']:.0f} chars) → insufficient signal")
            actions.append("merge adjacent chunks or add surrounding context")

        if r["uniqueness_mean"] < 0.2:
            issues.append(f"low lexical uniqueness ({r['uniqueness_mean']:.1%}) → vocabulary too shared")
            actions.append("add specific identifiers (ticker, fund name, date, section) to chunk text")

        if not issues:
            print(f"  {G}✓{RST}  {BOLD}{cat}{RST} — no structural issues detected")
            continue

        print(f"  {R}✗{RST}  {BOLD}{cat}{RST}")
        for issue in issues:
            print(f"       {Y}issue:{RST}  {issue}")
        for action in actions:
            print(f"       {G}fix:{RST}    {action}")
        print()


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]) -> None:
    print(f"\n{BOLD}{'═'*90}{RST}")
    print(f"{BOLD}  SUMMARY{RST}")
    print("─" * 90)
    header = f"  {'Category':30s} {'N':>4} {'AvgLen':>7} {'Unique%':>8} {'IntraSim':>9}  {'Retrievability':20}"
    print(f"{BOLD}{header}{RST}")
    print("  " + "─" * 86)
    for r in sorted(results, key=lambda x: x["mean_intra_sim"]):
        retrievability = 1.0 - r["mean_intra_sim"] * 0.5 - (1 - r["uniqueness_mean"]) * 0.5
        print(
            f"  {r['category']:30s} {r['count']:>4d} "
            f"{r['length_mean']:>7.0f} "
            f"{r['uniqueness_mean']:>7.1%} "
            f"{_colour(1 - r['mean_intra_sim'], (0.5, 0.7)):>9s}  "
            f"  {_bar(retrievability, width=20)}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"{BOLD}Loading corpus from Neo4j…{RST}")
    corpus = load_corpus_from_neo4j()
    print(f"Loaded {len(corpus)} chunks across "
          f"{len(set(c.category for c in corpus))} categories.\n")

    qa_pairs: List[Dict] = []
    if TEST_SET_PATH.exists():
        with open(TEST_SET_PATH) as f:
            qa_pairs = json.load(f)
        print(f"Loaded {len(qa_pairs)} QA pairs from cached test set.")
    else:
        print(f"{Y}No cached test set found — skipping query/chunk examples.{RST}")

    by_category: Dict[str, List[CorpusChunk]] = defaultdict(list)
    for chunk in corpus:
        by_category[chunk.category].append(chunk)

    results = []
    for cat, chunks in sorted(by_category.items()):
        r = analyse_category(cat, chunks, qa_pairs, verbose=True)
        results.append(r)

    print_summary(results)
    print_suggestions(results)


if __name__ == "__main__":
    main()
