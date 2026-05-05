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
"""

from __future__ import annotations

import logging
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
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CHUNK_QUERIES = {
    "RiskFactor_10K": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:RiskFactor)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title
        LIMIT $per_label
    """,
    "BusinessInformation": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:BusinessInformation)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title
        LIMIT $per_label
    """,
    "LegalProceeding": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:LegalProceeding)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title
        LIMIT $per_label
    """,
    "ManagementDiscussion": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:ManagementDiscussion)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title
        LIMIT $per_label
    """,
    "Properties": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:Properties)
        WHERE s.text IS NOT NULL AND size(s.text) > 200
        RETURN elementId(s) AS id, s.text AS text, s.title AS title
        LIMIT $per_label
    """,
    "Financials": """
        MATCH (f:Filing10K)-[:HAS_SECTION]->(s:Section:Financials)
        WHERE s.fiscalYear IS NOT NULL
          AND COALESCE(s.incomeStatement, s.balanceSheet, s.cashFlow, '') <> ''
        RETURN elementId(s) AS id,
               COALESCE(s.incomeStatement, s.balanceSheet, s.cashFlow) AS text
        LIMIT $per_label
    """,
    "Objective_Fund": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:Objective)
        WHERE s.text IS NOT NULL AND size(s.text) > 100
        RETURN elementId(s) AS id, s.text AS text, fund.name AS title
        LIMIT $per_label
    """,
    "PerformanceCommentary": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:PerformanceCommentary)
        WHERE s.text IS NOT NULL AND size(s.text) > 100
        RETURN elementId(s) AS id, s.text AS text, fund.name AS title
        LIMIT $per_label
    """,
    "RiskFactor_Fund": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:RiskFactor)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text IS NOT NULL AND size(c.text) > 50
        RETURN elementId(c) AS id, c.text AS text, fund.name AS title
        LIMIT $per_label
    """,
    "Strategy": """
        MATCH (fund:Fund)-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:Strategy)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text IS NOT NULL AND size(c.text) > 50
        RETURN elementId(c) AS id, c.text AS text, fund.name AS title
        LIMIT $per_label
    """,
}


def load_corpus_from_neo4j(
    per_label: int = 50,
    labels: Optional[List[str]] = None,
    max_chunk_chars: int = 4000,
) -> List[CorpusChunk]:
    """Sample chunks across the configured labels.

    Args:
        per_label: max chunks to fetch per label.
        labels:    restrict to a subset of labels (None => all configured).
        max_chunk_chars: hard truncation to keep encoding cost bounded.
    """
    selected = labels or list(_CHUNK_QUERIES.keys())
    chunks: List[CorpusChunk] = []

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
                    chunks.append(
                        CorpusChunk(
                            chunk_id=f"{label}:{raw_id}",
                            text=text,
                            category=label,
                            parent=row.get("title") or "",
                        )
                    )
                logger.info("Loaded %d chunks for label %s", len(rows), label)
    finally:
        db.close()

    return chunks
