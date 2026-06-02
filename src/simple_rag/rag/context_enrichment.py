"""
Context Enrichment — supplementary Cypher queries that auto-fire based on query conditions.

When the user asks about a specific fund or company, the main Text2Cypher query
may return only the directly requested data (e.g. expense ratio). The enrichment
engine detects the entities involved and runs static queries to fetch broader
context (full fund profile, provider, trust, managers, etc.) so the answer
generation LLM can produce richer, more informative responses.

Usage:
    enricher = ContextEnricher(neo4j_driver, entity_resolver)
    supplementary = enricher.enrich(category="fund_basic", user_query="What is VTI?")
    # supplementary is a dict of {rule_name: [rows]} with extra context
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ── Enrichment rule definition ───────────────────────────────────────────────

@dataclass
class EnrichmentRule:
    """A single enrichment rule: when it fires and what query to run."""
    name: str
    description: str
    categories: list[str]          # fires when the query category is one of these
    cypher_template: str           # Cypher with $ticker or $name parameters
    param_type: str                # "fund_ticker", "company_ticker", "provider_name", "none"
    priority: int = 0             # higher = runs first


# ── Static enrichment rules ──────────────────────────────────────────────────

ENRICHMENT_RULES: list[EnrichmentRule] = [
    EnrichmentRule(
        name="fund_overview",
        description="Full fund details with provider, trust, and latest financial highlights",
        categories=["fund_basic", "fund_portfolio", "fund_profile"],
        param_type="fund_ticker",
        priority=10,
        cypher_template="""
            MATCH (prov:Provider)-[:MANAGES]->(t:Trust)-[:ISSUES]->(f:Fund)
            WHERE f.ticker = $ticker
            OPTIONAL MATCH (f)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight)
            RETURN f.ticker AS ticker, f.name AS fundName,
                   f.securityExchange AS exchange,
                   prov.name AS provider, t.name AS trust,
                   fh.expenseRatio AS expenseRatio, fh.netAssets AS netAssets,
                   fh.turnover AS turnover, fh.advisoryFees AS advisoryFees,
                   fh.totalReturn AS totalReturn, fh.netIncomeRatio AS netIncomeRatio,
                   r.year AS year
            ORDER BY r.year DESC
            LIMIT 1
        """,
    ),
    
    EnrichmentRule(
        name="fund_returns",
        description="Average annual returns for the fund",
        categories=["fund_basic"],
        param_type="fund_ticker",
        priority=4,
        cypher_template="""
            MATCH (f:Fund)-[r:HAS_AVERAGE_RETURNS]->(ar:AverageReturns)
            WHERE f.ticker = $ticker
            RETURN ar.return1y AS return1y, ar.return5y AS return5y,
                   ar.return10y AS return10y, ar.returnInception AS returnInception,
                   r.date AS asOfDate
            ORDER BY r.date DESC
            LIMIT 1
        """,
    ),
]


# ── Enrichment engine ────────────────────────────────────────────────────────

class ContextEnricher:
    """
    Runs supplementary Cypher queries to enrich LLM context.

    Args:
        neo4j_driver:     Active Neo4j driver instance.
        entity_resolver:  EntityResolver instance (for extracting tickers/names from query).
    """

    def __init__(self, neo4j_driver, entity_resolver=None):
        self.driver = neo4j_driver
        self.entity_resolver = entity_resolver

    def enrich(
        self,
        category: str,
        user_query: str,
        main_results: Optional[list[dict]] = None,
    ) -> Dict[str, list[dict]]:
        """
        Run all applicable enrichment queries and return supplementary context.

        Args:
            category:      The classified query category (e.g. "fund_basic").
            user_query:    The original natural-language question.
            main_results:  Results from the primary Cypher query (used to extract
                           entities if entity_resolver is unavailable).

        Returns:
            Dict mapping rule names to their result rows, e.g.:
            {"fund_overview": [{"ticker": "VTI", "fundName": "...", ...}]}
        """
        # Determine which entities are in play
        params = self._extract_params(category, user_query, main_results)

        if not params:
            logger.debug("No entities detected for enrichment — skipping")
            return {}

        # Find applicable rules
        applicable = [
            rule for rule in ENRICHMENT_RULES
            if category in rule.categories and self._has_required_params(rule, params)
        ]
        applicable.sort(key=lambda r: r.priority, reverse=True)

        if not applicable:
            return {}

        # Execute enrichment queries
        supplementary: Dict[str, list[dict]] = {}
        for rule in applicable:
            rule_params = self._build_rule_params(rule, params)
            rows = self._execute(rule.cypher_template, rule_params)
            if rows:
                supplementary[rule.name] = rows
                logger.info(f"Enrichment [{rule.name}]: {len(rows)} row(s)")

        if supplementary:
            rule_names = ", ".join(supplementary.keys())
            print(f"🔍 Context enrichment: {rule_names}")

        return supplementary

    # ── Parameter extraction ─────────────────────────────────────────────

    def _extract_params(
        self,
        category: str,
        user_query: str,
        main_results: Optional[list[dict]],
    ) -> Dict[str, str]:
        """Extract entity parameters from the query and/or main results."""
        params: Dict[str, str] = {}

        # 1. Try entity resolver (most reliable)
        if self.entity_resolver:
            entities = self.entity_resolver.extract_entities(user_query)
            for name, info in entities.items():
                entity_type = info.get("type", "")
                if entity_type == "Ticker":
                    # Determine if it's a fund or company ticker based on category
                    if category.startswith("fund"):
                        params.setdefault("fund_ticker", name)
                    elif category.startswith("company"):
                        params.setdefault("company_ticker", name)
                    else:
                        params.setdefault("fund_ticker", name)
                elif entity_type == "Fund":
                    ticker = info.get("ticker")
                    if ticker:
                        params.setdefault("fund_ticker", ticker)
                elif entity_type == "Provider":
                    params.setdefault("provider_name", name)

        # 2. Fallback: extract tickers from main results
        if not params and main_results:
            for row in main_results[:5]:
                for key in ("ticker", "f.ticker", "c.ticker"):
                    val = row.get(key)
                    if val and isinstance(val, str):
                        if category.startswith("fund"):
                            params.setdefault("fund_ticker", val)
                        elif category.startswith("company"):
                            params.setdefault("company_ticker", val)
                        break

        # 3. Last resort: regex for uppercase tickers in the query
        if not params:
            ticker_match = re.findall(r'\b[A-Z]{2,5}\b', user_query)
            if ticker_match:
                ticker = ticker_match[0]
                if category.startswith("fund"):
                    params["fund_ticker"] = ticker
                elif category.startswith("company"):
                    params["company_ticker"] = ticker

        return params

    def _has_required_params(self, rule: EnrichmentRule, params: Dict[str, str]) -> bool:
        """Check if we have the parameter this rule needs."""
        if rule.param_type == "none":
            return True
        return rule.param_type in params

    def _build_rule_params(self, rule: EnrichmentRule, params: Dict[str, str]) -> Dict[str, str]:
        """Map extracted params to the Cypher template's $variables."""
        if rule.param_type == "fund_ticker":
            return {"ticker": params["fund_ticker"]}
        elif rule.param_type == "company_ticker":
            return {"ticker": params["company_ticker"]}
        elif rule.param_type == "provider_name":
            return {"name": params["provider_name"]}
        return {}

    # ── Query execution ──────────────────────────────────────────────────

    def _execute(self, cypher: str, params: Dict[str, str]) -> Optional[list[dict]]:
        """Run a Cypher query and return result rows."""
        try:
            with self.driver.session() as session:
                records = session.run(cypher, params)
                return [dict(record) for record in records]
        except Exception as e:
            logger.warning(f"Enrichment query failed: {e}")
            return None


# ── Format enrichment for prompt injection ────────────────────────────────────

def format_enrichment_context(supplementary: Dict[str, list[dict]]) -> str:
    """
    Format supplementary data into a text block suitable for injecting
    into the answer generation prompt.

    Returns an empty string if there is no enrichment data.
    """
    if not supplementary:
        return ""

    sections = []
    for rule_name, rows in supplementary.items():
        label = rule_name.replace("_", " ").title()
        if len(rows) == 1:
            # Single row: inline key-value pairs
            row = rows[0]
            pairs = [f"  {k}: {v}" for k, v in row.items() if v is not None]
            sections.append(f"[{label}]\n" + "\n".join(pairs))
        else:
            # Multiple rows: list format
            items = []
            for row in rows[:10]:  # cap enrichment rows
                cleaned = {k: v for k, v in row.items() if v is not None}
                items.append(f"  {cleaned}")
            sections.append(f"[{label}] ({len(rows)} records)\n" + "\n".join(items))

    return "\n\n".join(sections)


# ── Document provenance resolver ──────────────────────────────────────────────
#
# Inspects the generated Cypher to detect which node labels are queried, then
# builds and runs a supplementary query to fetch the source Document — but only
# if the Cypher doesn't already include Document / EXTRACTED_FROM / DISCLOSED_IN.

# Maps node labels to their path to the nearest Document node.
# Maps the ANCHOR label detected in a Cypher query to its document source path.
#
# Only anchor nodes are listed here — child nodes are always queried alongside
# their parent anchor, so the anchor's detection is sufficient:
#
#   Fund anchor     → covers: FinancialHighlight, AverageReturns, ShareClass,
#                             Sector, Region, Table, Trust, Provider
#   Profile anchor  → covers: Section:Objective, Section:Strategy,
#                             Section:RiskFactor, Section:PerformanceCommentary,
#                             and all Chunk nodes under those sections
#   Portfolio anchor→ covers: Portfolio properties and Holding nodes
#   Filing10K anchor→ covers: all 10-K Section types and their Chunk nodes
#
_LABEL_TO_DOCUMENT: Dict[str, tuple[str, str]] = {
    "Fund": (
        "(f:Fund)-[:EXTRACTED_FROM]->(d:Document) WHERE f.ticker = $id",
        "fund_ticker",
    ),
    "Profile": (
        "(f:Fund)-[:DEFINED_BY]->(pr:Profile)-[:EXTRACTED_FROM]->(d:Document) WHERE f.ticker = $id",
        "fund_ticker",
    ),
    "Portfolio": (
        "(f:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio)-[:EXTRACTED_FROM]->(d:Document) WHERE f.ticker = $id",
        "fund_ticker",
    ),
    "Filing10K": (
        "(c:Company)-[:REPORTS_IN]->(fk:Filing10K)-[:EXTRACTED_FROM]->(d:Document) WHERE c.ticker = $id",
        "company_ticker",
    ),
    "InsiderTransaction": (
        "(c:Company)-[:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction)-[:EXTRACTED_FROM]->(d:Document) WHERE c.ticker = $id",
        "company_ticker",
    ),
    "CompensationPackage": (
        "(c:Company)-[:HAS_CEO]->(:Person)-[:RECEIVED_COMPENSATION]->(cp:CompensationPackage)-[:DISCLOSED_IN]->(d:Document) WHERE c.ticker = $id",
        "company_ticker",
    ),
}

# Walk order for collecting unique document sources.
# Company-specific anchors before fund anchors so the correct identifier type
# is used when both appear in the same query (rare, but safe).
_LABEL_PRIORITY = [
    "CompensationPackage",
    "InsiderTransaction",
    "Filing10K",
    "Portfolio",
    "Profile",
    "Fund",
]

# Regex to detect node labels in Cypher — matches :Label patterns
_LABEL_RE = re.compile(r':([A-Z][a-zA-Z0-9]+)')
# Regex to detect if the Cypher already fetches Document info
_ALREADY_HAS_DOC_RE = re.compile(
    r':Document|EXTRACTED_FROM|DISCLOSED_IN', re.IGNORECASE
)
# Regex to extract a ticker literal from the Cypher
_TICKER_LITERAL_RE = re.compile(r"""ticker\s*[:=]\s*['"]([A-Za-z]+)['"]""")
# Regex to extract a company name literal from the Cypher
_NAME_LITERAL_RE = re.compile(r"""name\s*[:=]\s*['"]([^'"]+)['"]""")


def _extract_identifier_from_cypher(cypher: str) -> Optional[str]:
    """Try to pull a ticker or name literal from the generated Cypher."""
    m = _TICKER_LITERAL_RE.search(cypher)
    if m:
        return m.group(1)
    return None


def _extract_identifier_from_results(results: Optional[list[dict]]) -> Optional[str]:
    """Pull a ticker from the first row of query results."""
    if not results:
        return None
    for row in results[:3]:
        for key in ("ticker", "f.ticker", "c.ticker", "fundTicker", "companyTicker"):
            val = row.get(key)
            if val and isinstance(val, str):
                return val
    return None


def resolve_document_provenance(
    cypher: str,
    neo4j_driver,
    main_results: Optional[list[dict]] = None,
) -> str:
    """
    Inspect a generated Cypher query, detect which nodes it touches, and —
    if it doesn't already include Document info — run a supplementary query
    per unique document source to fetch all relevant source documents.

    A single query may touch nodes backed by different documents (e.g. a
    full fund overview that joins Fund, Profile, and Portfolio — each
    EXTRACTED_FROM a different SEC filing).  This function collects all
    unique document-source paths found in the Cypher and returns one
    provenance record per distinct path.

    Returns a formatted provenance string (empty if not applicable).
    """
    if not cypher:
        return ""

    # 1. Skip if the Cypher already references Document / EXTRACTED_FROM
    if _ALREADY_HAS_DOC_RE.search(cypher):
        logger.debug("Cypher already includes Document — skipping provenance")
        return ""

    # 2. Detect node labels in the Cypher
    labels_in_cypher = set(_LABEL_RE.findall(cypher))
    if not labels_in_cypher:
        return ""

    # 3. Collect ALL unique document-source paths (not just the first match).
    #    Walk the priority list so that when two labels share the same path,
    #    the more-specific label's name is used for the section header.
    seen_fragments: set[str] = set()
    sources: list[tuple[str, str]] = []   # [(label_name, match_fragment), ...]
    for label in _LABEL_PRIORITY:
        if label in labels_in_cypher and label in _LABEL_TO_DOCUMENT:
            match_fragment, _id_type = _LABEL_TO_DOCUMENT[label]
            if match_fragment not in seen_fragments:
                seen_fragments.add(match_fragment)
                sources.append((label, match_fragment))

    if not sources:
        return ""

    # 4. Extract identifier (ticker) from the Cypher or from results
    identifier = _extract_identifier_from_cypher(cypher)
    if not identifier:
        identifier = _extract_identifier_from_results(main_results)
    if not identifier:
        logger.debug("No identifier found for provenance query")
        return ""

    # 5. Run one query per unique document source and collect results
    all_docs: list[str] = []
    for label_name, match_fragment in sources:
        doc_cypher = (
            f"MATCH {match_fragment}\n"
            f"RETURN d.accessionNumber AS accessionNumber, d.url AS documentUrl,\n"
            f"       d.type AS documentType, d.filingDate AS filingDate,\n"
            f"       d.reportingDate AS reportingDate\n"
            f"ORDER BY d.filingDate DESC LIMIT 1"
        )
        try:
            with neo4j_driver.session() as session:
                records = session.run(doc_cypher, {"id": identifier})
                rows = [dict(r) for r in records]
        except Exception as e:
            logger.warning(f"Provenance query failed for {label_name}: {e}")
            continue

        for row in rows:
            parts: list[str] = []
            if row.get("accessionNumber"):
                parts.append(f"  Accession Number: {row['accessionNumber']}")
            if row.get("documentUrl"):
                parts.append(f"  URL: {row['documentUrl']}")
            if row.get("documentType"):
                parts.append(f"  Filing Type: {row['documentType']}")
            if row.get("filingDate"):
                parts.append(f"  Filing Date: {row['filingDate']}")
            if row.get("reportingDate"):
                parts.append(f"  Reporting Date: {row['reportingDate']}")
            if parts:
                source_label = label_name.replace("_", " ")
                all_docs.append(f"  [{source_label}]\n" + "\n".join(parts))

    if not all_docs:
        return ""

    return "[Source Documents]\n" + "\n---\n".join(all_docs)
