"""Pull a sample text corpus from the Neo4j knowledge graph.

The benchmark needs a heterogeneous chunk pool to test retrieval. We
sample across section/chunk nodes across both 10-K filings and fund
profile documents:

Company filings (10-K):
    Section:RiskFactor, Section:BusinessInformation, Section:LegalProceeding,
    Section:ManagementDiscussion, Section:Properties, Section:Financials

Fund profiles (Profile):
    Section:Objective, Section:PerformanceCommentary,
    Section:RiskFactor (fund-level), Section:Strategy

Each chunk gets a stable id, the text, the category, and a parent title.
Duplicate source texts (same content across multiple Neo4j nodes, e.g. the
same fund section repeated across share classes) are removed before the
corpus is returned.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from simple_rag.database.neo4j.neo4j import Neo4jDatabase

logger = logging.getLogger(__name__)


@dataclass
class CorpusChunk:
    chunk_id: str
    text: str
    category: str
    parent: str = ""
    doc_id: str = ""
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CHUNK_QUERIES = {
    "RiskFactor_10K": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:RiskFactor)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title,
               elementId(f) AS doc_id
        LIMIT $per_label
    """,
    "BusinessInformation": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:BusinessInformation)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title,
               elementId(f) AS doc_id
        LIMIT $per_label
    """,
    "LegalProceeding": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:LegalProceeding)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title,
               elementId(f) AS doc_id
        LIMIT $per_label
    """,
    "ManagementDiscussion": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:ManagementDiscussion)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title,
               elementId(f) AS doc_id
        LIMIT $per_label
    """,
    "Properties": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:Properties)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title,
               elementId(f) AS doc_id
        LIMIT $per_label
    """,
    "Financials": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:Financials)
        WHERE s.fiscalYear IS NOT NULL
          AND COALESCE(s.incomeStatement, s.balanceSheet, s.cashFlow, '') <> ''
        RETURN elementId(s) AS id,
               COALESCE(s.incomeStatement, s.balanceSheet, s.cashFlow) AS text,
               elementId(f) AS doc_id
        LIMIT $per_label
    """,
    "Objective_Fund": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:Objective)
        WHERE s.text IS NOT NULL AND size(s.text) > 100
        RETURN elementId(s) AS id, s.text AS text, fund.name AS title,
               elementId(p) AS doc_id
        LIMIT $per_label
    """,
    "PerformanceCommentary": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:PerformanceCommentary)
        WHERE s.text IS NOT NULL AND size(s.text) > 100
        RETURN elementId(s) AS id, s.text AS text, fund.name AS title,
               elementId(p) AS doc_id
        LIMIT $per_label
    """,
    "RiskFactor_Fund": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:RiskFactor)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text IS NOT NULL AND size(c.text) > 50
        RETURN elementId(c) AS id, c.text AS text, fund.name AS title,
               elementId(p) AS doc_id
        LIMIT $per_label
    """,
    "Strategy": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:Strategy)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text IS NOT NULL AND size(c.text) > 50
        RETURN elementId(c) AS id, c.text AS text, fund.name AS title,
               elementId(p) AS doc_id
        LIMIT $per_label
    """,
}


# Fund categories whose section text never contains the fund name.
# For these we prepend "Fund: <name>\n\n" so the embedding is identifiable.
_FUND_CATEGORIES = {
    "Objective_Fund",
    "PerformanceCommentary",
    "RiskFactor_Fund",
    "Strategy",
}


def _enrich_with_parent(chunks: List[CorpusChunk]) -> List[CorpusChunk]:
    """Prepend the fund name to chunks where it is not already in the text.

    Fund profile sections often use "the Fund" throughout without naming
    the fund.  Injecting the name makes each chunk identifiable by an
    embedding model and mirrors what a production chunker should do.
    """
    result = []
    for c in chunks:
        if c.category in _FUND_CATEGORIES and c.parent and c.parent not in c.text:
            c = CorpusChunk(
                chunk_id=c.chunk_id,
                text=f"Fund: {c.parent}\n\n{c.text}",
                category=c.category,
                parent=c.parent,
                doc_id=c.doc_id,
                extra=c.extra,
            )
        result.append(c)
    return result


def _deduplicate_and_diversify(
    chunks: List[CorpusChunk],
    min_docs: int = 5,
    seed: int = 42,
) -> List[CorpusChunk]:
    """Remove duplicate source texts and enforce document diversity.

    Within each category we:
    1. Shuffle (for random selection among ties).
    2. Keep the first chunk per unique (category, text) pair — dropping all
       later nodes that carry the same content.
    3. Warn if fewer than ``min_docs`` distinct documents (doc_id / parent)
       remain after deduplication.
    """
    rng = random.Random(seed)
    shuffled = chunks[:]
    rng.shuffle(shuffled)

    seen: set[tuple[str, str]] = set()
    result: List[CorpusChunk] = []
    for chunk in shuffled:
        key = (chunk.category, chunk.text)
        if key not in seen:
            seen.add(key)
            result.append(chunk)

    # Report document diversity per category
    docs_per_cat: Dict[str, set] = defaultdict(set)
    for c in result:
        docs_per_cat[c.category].add(c.doc_id or c.parent)

    for cat, doc_ids in sorted(docs_per_cat.items()):
        n = len(doc_ids)
        if n < min_docs:
            logger.warning(
                "Category '%s' has only %d distinct document(s) after dedup "
                "(wanted >= %d). Ingest more documents for reliable benchmarks.",
                cat, n, min_docs,
            )
        else:
            logger.info("Category '%s': %d distinct documents after dedup.", cat, n)

    return result


def load_corpus_from_neo4j(
    per_label: int = 50,
    labels: Optional[List[str]] = None,
    max_chunk_chars: int = 4000,
    min_docs: int = 5,
    seed: int = 42,
) -> List[CorpusChunk]:
    """Sample chunks across the configured labels.

    Args:
        per_label:      max chunks to fetch per label (before dedup).
        labels:         restrict to a subset of labels (None => all configured).
        max_chunk_chars: hard truncation to keep encoding cost bounded.
        min_docs:       minimum distinct documents expected per category;
                        logs a warning if not met after deduplication.
        seed:           RNG seed used when randomly selecting among duplicate texts.
    """
    selected = labels or list(_CHUNK_QUERIES.keys())
    raw_chunks: List[CorpusChunk] = []

    db = Neo4jDatabase()
    try:
        with db.driver.session() as session:
            for label in selected:
                query = _CHUNK_QUERIES.get(label)
                if not query:
                    logger.warning("Unknown label %s, skipping", label)
                    continue

                try:
                    result = session.run(query, per_label=per_label)
                    rows = list(result)
                except Exception as e:
                    logger.warning("Query for label %s failed: %s", label, e)
                    continue

                for row in rows:
                    raw_id = row.get("id")
                    text = row.get("text") or ""
                    if not raw_id or not text:
                        continue
                    text = text.strip()
                    if len(text) > max_chunk_chars:
                        text = text[:max_chunk_chars]
                    raw_chunks.append(
                        CorpusChunk(
                            chunk_id=f"{label}:{raw_id}",
                            text=text,
                            category=label,
                            parent=row.get("title") or "",
                            doc_id=str(row.get("doc_id") or ""),
                        )
                    )
                logger.info("Fetched %d raw chunks for label %s", len(rows), label)
    finally:
        db.close()

    enriched = _enrich_with_parent(raw_chunks)
    chunks = _deduplicate_and_diversify(enriched, min_docs=min_docs, seed=seed)
    logger.info(
        "Corpus ready: %d unique chunks from %d raw (removed %d duplicates)",
        len(chunks), len(raw_chunks), len(raw_chunks) - len(chunks),
    )
    return chunks
