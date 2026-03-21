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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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
        description="Full fund details with provider and trust",
        categories=["fund_basic", "fund_portfolio", "fund_profile"],
        param_type="fund_ticker",
        priority=10,
        cypher_template="""
            MATCH (prov:Provider)-[:MANAGES]->(t:Trust)-[:ISSUES]->(f:Fund)
            WHERE f.ticker = $ticker
            RETURN f.ticker AS ticker, f.name AS fundName,
                   f.expenseRatio AS expenseRatio, f.netAssets AS netAssets,
                   f.turnoverRate AS turnoverRate, f.advisoryFees AS advisoryFees,
                   f.numberHoldings AS numberHoldings, f.costsPer10k AS costsPer10k,
                   f.securityExchange AS exchange,
                   prov.name AS provider, t.name AS trust
        """,
    ),
    EnrichmentRule(
        name="fund_managers",
        description="Fund manager(s) with date",
        categories=["fund_basic", "fund_portfolio", "fund_profile"],
        param_type="fund_ticker",
        priority=5,
        cypher_template="""
            MATCH (f:Fund)-[r:MANAGED_BY]->(p:Person)
            WHERE f.ticker = $ticker
            RETURN p.name AS manager, r.date AS sinceDate
            ORDER BY r.date DESC
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
    EnrichmentRule(
        name="company_overview",
        description="Basic company information",
        categories=["company_filing", "company_people"],
        param_type="company_ticker",
        priority=10,
        cypher_template="""
            MATCH (c:Company)
            WHERE c.ticker = $ticker
            OPTIONAL MATCH (c)-[:HAS_CEO]->(ceo:Person)
            RETURN c.ticker AS ticker, c.name AS companyName, c.cik AS cik,
                   ceo.name AS ceo
            LIMIT 1
        """,
    ),
    EnrichmentRule(
        name="company_latest_filing",
        description="Most recent 10-K filing date and document",
        categories=["company_filing"],
        param_type="company_ticker",
        priority=5,
        cypher_template="""
            MATCH (c:Company)-[r:HAS_FILING]->(f:Filing10K)-[:EXTRACTED_FROM]->(d:Document)
            WHERE c.ticker = $ticker
            RETURN c.ticker AS ticker, r.date AS filingDate,
                   d.url AS documentUrl, d.accessionNumber AS accessionNumber
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
