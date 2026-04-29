"""
Query Handler — orchestrates classification → schema selection → Text2Cypher / vector search.

This module is the single entry-point for answering a user query.
It uses the trained SetFit classifier to decide which *schema slice*
to feed to the Text2Cypher LLM, drastically reducing prompt size
and improving accuracy.

Usage:
    from simple_rag.rag.query_handler import QueryHandler

    handler = QueryHandler(neo4j_driver=driver)
    result  = handler.handle("What is the expense ratio of VTI?")
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from simple_rag.rag.query.query_classification import QueryClassifier, QueryCategory, LABELS
from simple_rag.rag.text2cypher import CypherTranslator
from simple_rag.rag.schema_definitions import DETAILED_SCHEMA
from simple_rag.rag.context_enrichment import ContextEnricher

logger = logging.getLogger(__name__)

# ─── Schema slices ────────────────────────────────────────────────────────────
# Each slice contains ONLY the subgraph relevant to its category,
# keeping the LLM prompt small and focused.

from typing import Dict

SCHEMA_SLICES: Dict[str, str] = {

    "fund_basic": """
Relevant schema: 
(:Provider /* properties: name */)-[:MANAGES]->(:Trust /* properties: name */)-[:ISSUES]->(f:Fund /* properties: ticker, name, expenseRatio, netAssets, turnoverRate, advisoryFees, numberHoldings, costsPer10k, securityExchange */)
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass /* properties: name, description */)
(:Fund)-[:HAS_CHART]->(:Image /* properties: category, svg, title */)
(:Fund)-[:EXTRACTED_FROM]->(:Document /* properties: url, type, filingDate, accessionNumber */)
(:Fund)-[:HAS_AVERAGE_RETURNS /* properties: date */]->(:AverageReturns /* properties: return1y, return5y, return10y, returnInception */)
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT /* properties: year */]->(:FinancialHighlight /* properties: turnover, expenseRatio, totalReturn, netAssets, netAssetsValueBeginning, netAssetsValueEnd, netIncomeRatio */)
(:Fund)-[:EXTRACTED_FROM]->(:Document /* properties: url, type, filingDate, reportingDate, accessionNumber */)
(:Fund)-[r:HAS_SECTOR_ALLOCATION /* properties: weight, date */]->(:Sector /* properties: name */)
(:Fund)-[r:HAS_REGION_ALLOCATION /* properties: weight, date */]->(:Region /* properties: name */)
QUERY RULES:
- For name searches use CALL db.index.fulltext.queryNodes('fundNameIndex', 'search_term')
- For exact ticker matching use {ticker: 'VTI'} directly.
- numberHoldings is pre-calculated — do NOT count holdings.
- The year is on the RELATIONSHIP, not the node: use r.year.
- turnover is absolute (2 = 2%, not 0.02).
- Use the numberHoldings property of the Fund node to get the number of holdings.
- ALWAYS return the source of the information via the EXTRACTED FROM document node.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. STRICT SCHEMA ALIGNMENT: `netAssets` and `turnoverRate` are DIRECTLY on the `Fund` node. Do NOT look for them on Portfolio or FinancialHighlight nodes. `turnoverRate` is absolute (2 = 2%, not 0.02).
2. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
3. COMPARING ENTITIES: When asked to compare multiple funds (e.g., "Compare VTI and VOO"), DO NOT match them into a single row. Return each fund as a separate row using `IN` (e.g., `WHERE f.ticker IN ['VTI', 'VOO'] RETURN f.ticker...`).
4. LATEST / LAST DATA: When a user asks for the "last", "latest", or "current" metric, ALWAYS use `ORDER BY r.year DESC LIMIT 1`.
5. HISTORICAL GROWTH (Since X years ago): To calculate growth over time, MATCH the highlights, order by year, `collect(fh)`, and compare `highlights[0]` (latest) to `highlights[X]` (previous). 
6. DIVISION BY ZERO: When calculating percentages, ALWAYS use `CASE WHEN denominator = 0 THEN 0 ELSE (numerator * 100.0 / denominator) END`.
7. INCOMPLETE FILTERS: NEVER generate empty or incomplete property filters like `(n:Label {name})`. Only include explicit values like `{name: 'Vanguard'}`.
""",

    "fund_portfolio": """
Relevant schema:
(:Fund /* properties: ticker, name */)-[:HAS_PORTFOLIO]->(p:Portfolio /* properties: date, seriesId */)
(p)-[:HAS_HOLDING /* properties: shares, marketValue, weight, fairValueLevel, isRestricted, payoffProfile */]->(h:Holding /* properties: name, ticker, isin, lei, category, country, sector, assetCategory, issuerCategory */)
(:Holding)-[:REPRESENTS]->(:Company /* properties: ticker, name */)
(p)-[:EXTRACTED_FROM]->(:Document /* properties: url, type, filingDate, reportingDate, accessionNumber */)
QUERY RULES:
- Use f.numberHoldings for count — do NOT count holdings manually.
- Weight on HAS_HOLDING is the portfolio weight.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. STRICT SCHEMA ALIGNMENT: `payoffProfile`, `marketValue`, and `weight` are properties of the `[r:HAS_HOLDING]` relationship, NOT the `Holding` node. 
2. INLINE MATH: NEVER use math operators (`>`, `<`, `gt`) inside curly braces in MATCH patterns (e.g., BAD: `[r:HAS_HOLDING {marketValue: > 10000}]`). ALWAYS use a `WHERE` clause (e.g., `WHERE r.marketValue > 10000`).
3. DIVISION BY ZERO: When calculating percentage ratios, ALWAYS use `CASE WHEN total = 0 THEN 0 ELSE (part * 100.0 / total) END`.
4. COMMAS: Ensure proper commas are placed between variables in the `RETURN` statement before aggregate functions (e.g., `RETURN a, COUNT(b)`).
""",

    "fund_profile": """
Contains all the general information of the fund, like the objective the risks, the performance commentary, etc:
(:Fund /* properties: ticker, name */)-[:DEFINED_BY /* properties: date */]->(:Profile /* properties: id, summaryProspectus */)
(:Profile)-[:HAS_STRATEGY_CHUNK]->(:StrategyChunk /* properties: id, title, text, embedding */)
(:Profile)-[:HAS_RISK_CHUNK]->(:RiskChunk /* properties: id, title, text, embedding */)
(:Profile)-[:HAS_OBJECTIVE_CHUNK]->(:Objective /* properties: id, text, embedding */)
(:Profile)-[:HAS_PERFORMANCE_COMMENTARY_CHUNK]->(:PerformanceCommentary /* properties: id, text, embedding */)
(:Profile)-[:EXTRACTED_FROM]->(:Document /* properties: accessionNumber, url,reportingDate, filingDate */)

QUERY RULES:
- These nodes have embeddings — use vector search for semantic queries.
- Vector indexes exist on embedding properties.
- ALWAYS return ticker and name alongside the requested data.
- Do NOT attempt to extract netAssets or numerical performance from these text nodes.
""",

    "company_filing": """
Relevant schema:
(:Company /* properties: ticker, name, cik */)-[:REPORTS_IN /* properties: year */]->(filing:Filing10K)
(filing)-[:HAS_SECTION]->(:Section:RiskFactor /* properties: title, text, sectionType, secItem */)
(filing)-[:HAS_SECTION]->(:Section:BusinessInformation /* properties: title, text, sectionType, secItem */)
(filing)-[:HAS_SECTION]->(:Section:ManagementDiscussion /* properties: title, text, sectionType, secItem */)
(filing)-[:HAS_SECTION]->(:Section:LegalProceeding /* properties: title, text, sectionType, secItem */)
(filing)-[:HAS_SECTION]->(:Section:Properties /* properties: title, text, sectionType, secItem */)
(:Section)-[:HAS_CHUNK]->(:Chunk:SectionChunk /* properties: title, text, embedding, chunkType */)
(filing)-[:HAS_FINACIALS]->(fin:Section:Financials /* properties: incomeStatement, balanceSheet, cashFlow, fiscalYear */)
(fin)-[:HAS_METRIC]->(:FinancialMetric /* properties: label, value */)
(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment /* properties: label, value, percentage */)
(filing)-[:EXTRACTED_FROM]->(:Document /* properties: accession_number, url, filing_date */)

QUERY RULES:
- Company ticker is a stock ticker like 'AAPL', 'MSFT'.
- The relationship is HAS_FINACIALS (with one 'N').

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
2. LATEST FILING: When asked for the "latest" or "newest" filing/metrics, use `ORDER BY filing.date DESC LIMIT 1`.
""",

    "company_people": """
Relevant schema:
(:Fund /* properties: ticker, name */)-[:MANAGED_BY /* properties: date */]->(:Person /* properties: name */)
(:Company /* properties: ticker, name, cik */)-[:HAS_CEO /* properties: ceoCompensation, ceoActuallyPaid, date */]->(:Person /* properties: name */)
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction /* properties: position, transactionType, shares, price, value, remainingShares */)-[:MADE_BY]->(:Person /* properties: name */)
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage /* properties: totalCompensation, shareholderReturn, date */)
(:CompensationPackage)-[:AWARDED_BY]->(:Company)

QUERY RULES:
- Fund managers are linked via MANAGED_BY.
- Company CEOs are linked via HAS_CEO.
- Use personNameIndex for fuzzy person name search.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
2. INLINE MATH: NEVER use math operators (`>`, `<`) inside node patterns. Use `WHERE` clauses.
""",

}


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Container for the full pipeline result."""
    query: str
    category: str
    confidence: float
    schema_used: str
    cypher: Optional[str] = None
    data: Any = None
    error: Optional[str] = None
    requires_vector_search: bool = False
    enrichment: Dict[str, Any] = field(default_factory=dict)


# ─── Handler ──────────────────────────────────────────────────────────────────

class QueryHandler:
    """
    Orchestrates: classify → pick schema → translate → (optionally execute).

    Args:
        neo4j_driver:     Active Neo4j driver instance.
        classifier_path:  Path to the trained SetFit model directory.
                          Defaults to ``query/models/query_classifier``
                          relative to this file.
        cypher_backend:   Backend for CypherTranslator ('ollama', 'groq', 'huggingface').
        cypher_model:     Model name for the CypherTranslator.
        confidence_threshold: Minimum confidence to trust the classifier.
                              Below this the handler falls back to ``cross_entity``
                              (the widest schema).
        **cypher_kwargs:  Extra keyword arguments forwarded to CypherTranslator.
    """

    

    def __init__(
        self,
        neo4j_driver,
        classifier_path: Optional[str] = None,
        cypher_backend: str = "ollama",
        cypher_model: str = "llama3.1:8b",
        confidence_threshold: float = 0.45,
        **cypher_kwargs,
    ):
        # ── Classifier ──────────────────────────────────────────────
        if classifier_path is None:
            classifier_path = str(
                Path(__file__).parent / "query" / "models" / "query_classifier"
            )
        self.classifier = QueryClassifier(classifier_path)

        # ── Text2Cypher translator ──────────────────────────────────
        self.translator = CypherTranslator(
            neo4j_driver=neo4j_driver,
            model_name=cypher_model,
            backend=cypher_backend,
            **cypher_kwargs,
        )

        self.neo4j_driver = neo4j_driver
        self.confidence_threshold = confidence_threshold

        # Context enrichment engine (reuses the translator's entity resolver)
        self.enricher = ContextEnricher(
            neo4j_driver=neo4j_driver,
            entity_resolver=self.translator.entity_resolver,
        )

        print("✓ QueryHandler ready")

    # ── public API ────────────────────────────────────────────────────────

    def handle(
        self,
        user_query: str,
        execute: bool = False,
        temperature: Optional[float] = None,
        use_schema_injection: bool = True,
    ) -> QueryResult:
        """
        Full pipeline: classify → select schema → translate → optionally execute.

        Args:
            user_query:           Natural-language question.
            execute:              If True, run the generated Cypher against Neo4j
                                  and attach the rows to ``result.data``.
            temperature:          Override temperature for the Text2Cypher LLM.
            use_schema_injection: If True (default), the query classifier selects a
                                  focused schema slice for the initial LLM call.
                                  If False, the full DETAILED_SCHEMA is used from
                                  the start (classification still runs for routing,
                                  but the schema sent to the LLM is always complete).
                                  Note: retries always use the full schema regardless.

        Returns:
            QueryResult with category, cypher, and optional data.
        """
        import time
        start_pipeline = time.time()
        
        # Step 1 — classify
        start_classification = time.time()
        prediction = self.classifier.predict(user_query)
        classification_time = time.time() - start_classification

        active_labels: list = prediction["labels"]          # e.g. ["fund_basic", "fund_portfolio"]
        top_label: str       = prediction["top_label"]       # highest-confidence label
        confidence: float    = prediction["confidence"]
        per_conf: dict       = prediction["per_label_confidence"]

        # Display
        label_str = " | ".join(
            f"{l} ({per_conf[l]:.2%})" for l in active_labels
        )
        logger.info(f"Classification: {label_str}")
        print(f"🏷️  Active labels: {label_str}")

        # Step 2 — pick schema slice(s)
        if QueryCategory.NOT_RELATED.value in active_labels:
            result = QueryResult(
                query=user_query, category=QueryCategory.NOT_RELATED.value,
                confidence=confidence, schema_used="",
            )
            result.error = "Query is not related to the financial database."
            print("🚫 Not related — skipping.")
            return result

        if len(active_labels) == 1:
            selected_schema_name = active_labels[0]
            schema_slice = SCHEMA_SLICES.get(active_labels[0], "")
            print(f"→ Single label: {active_labels[0]} ({confidence:.2%})")
        else:
            # Merge all active schema slices
            selected_schema_name = " + ".join(active_labels)
            parts = []
            for lbl in active_labels:
                s = SCHEMA_SLICES.get(lbl, "")
                if s:
                    parts.append(f"=== Schema: {lbl} ===\n{s}")
            schema_slice = "\n\n".join(parts)
            print(f"🔀 Multi-label: [{', '.join(active_labels)}] — merging schemas")

        # Step 3 — build result container
        result = QueryResult(
            query=user_query,
            category=top_label,
            confidence=confidence,
            schema_used=selected_schema_name,
        )

        # fund_profile → also flag vector search
        if QueryCategory.FUND_PROFILE.value in active_labels:
            result.requires_vector_search = True
            print("🔀 Hybrid — will generate Cypher AND flag vector search.")

        # Step 4 — translate with the sliced schema (or full schema if injection disabled)
        if use_schema_injection:
            effective_schema = schema_slice
            print(f"📐 Schema: slice ({selected_schema_name})")
        else:
            effective_schema = DETAILED_SCHEMA
            print("📐 Schema: full DETAILED_SCHEMA (injection disabled)")
        start_translation = time.time()
        cypher = self._translate_with_schema(
            user_query, effective_schema, temperature=temperature
        )
        translation_time = time.time() - start_translation
        result.cypher = cypher

        if cypher is None:
            result.error = "Failed to generate Cypher query."
            return result

        # Step 5 — optionally execute
        if execute and cypher:
            result.data = self._execute(cypher)

        # Step 6 — context enrichment (supplementary queries for richer LLM context)
        if execute and result.data is not None:
            result.enrichment = self.enricher.enrich(
                category=top_label,
                user_query=user_query,
                main_results=result.data,
            )

        total_pipeline_time = time.time() - start_pipeline
        print(f"\n⏱️  Pipeline timings:")
        print(f"   - Classification (CPU): {classification_time:.3f}s")
        print(f"   - Translation (LLM):    {translation_time:.3f}s")
        print(f"   - Total Pipeline Time:  {total_pipeline_time:.3f}s")

        return result

    # ── internals ─────────────────────────────────────────────────────────

    def _translate_with_schema(
        self,
        user_query: str,
        schema_slice: str,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """
        Temporarily swap the translator's schema, call translate(),
        then restore the original schema.
        """
        original_schema = self.translator.schema
        try:
            self.translator.schema = schema_slice
            return self.translator.translate(user_query, temperature=temperature)
        finally:
            self.translator.schema = original_schema

    def _execute(self, cypher: str) -> Any:
        """Run a Cypher query and return the result rows."""
        try:
            with self.neo4j_driver.session() as session:
                records = session.run(cypher)
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            print(f"❌ Execution error: {e}")
            return None
