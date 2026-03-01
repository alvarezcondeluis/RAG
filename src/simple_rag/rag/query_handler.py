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

QUERY RULES:
- For name searches use CALL db.index.fulltext.queryNodes('fundNameIndex', 'search_term')
- For exact ticker matching use {ticker: 'VTI'} directly.
- numberHoldings is pre-calculated — do NOT count holdings.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. STRICT SCHEMA ALIGNMENT: `netAssets` and `turnoverRate` are DIRECTLY on the `Fund` node. Do NOT look for them on Portfolio or FinancialHighlight nodes. `turnoverRate` is absolute (2 = 2%, not 0.02).
2. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
3. COMPARING ENTITIES: When asked to compare multiple funds (e.g., "Compare VTI and VOO"), DO NOT match them into a single row. Return each fund as a separate row using `IN` (e.g., `WHERE f.ticker IN ['VTI', 'VOO'] RETURN f.ticker...`).
4. INCOMPLETE FILTERS: NEVER generate empty or incomplete property filters like `(n:Label {name})`. Only include explicit values like `{name: 'Vanguard'}`.
""",

    "fund_performance": """
Relevant schema:
(:Fund /* properties: ticker, name */)-[:HAS_TRAILING_PERFORMANCE /* properties: date */]->(:TrailingPerformance /* properties: return1y, return5y, return10y, returnInception */)
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT /* properties: year */]->(:FinancialHighlight /* properties: turnover, expenseRatio, totalReturn, netAssets, netAssetsValueBeginning, netAssetsValueEnd, netIncomeRatio */)

QUERY RULES:
- The year is on the RELATIONSHIP, not the node: use r.year.
- turnover is absolute (2 = 2%, not 0.02).
- ALWAYS return fund ticker and name alongside the requested data.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. LATEST / LAST DATA: When a user asks for the "last", "latest", or "current" metric, ALWAYS use `ORDER BY r.year DESC LIMIT 1`.
2. HISTORICAL GROWTH (Since X years ago): To calculate growth over time, MATCH the highlights, order by year, `collect(fh)`, and compare `highlights[0]` (latest) to `highlights[X]` (previous). 
3. DIVISION BY ZERO: When calculating percentages, ALWAYS use `CASE WHEN denominator = 0 THEN 0 ELSE (numerator * 100.0 / denominator) END`.
4. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
""",

    "fund_portfolio": """
Relevant schema:
(:Fund /* properties: ticker, name */)-[:HAS_PORTFOLIO]->(p:Portfolio /* properties: date, seriesId */)
(p)-[:HAS_HOLDING /* properties: shares, marketValue, weight, fairValueLevel, isRestricted, payoffProfile */]->(h:Holding /* properties: name, ticker, isin, lei, category, country, sector, assetCategory, issuerCategory */)
(:Holding)-[:REPRESENTS]->(:Company /* properties: ticker, name */)
(:Fund)-[:HAS_SECTOR_ALLOCATION /* properties: weight, date */]->(:Sector /* properties: name */)
(:Fund)-[:HAS_GEOGRAPHIC_ALLOCATION /* properties: weight, date */]->(:GeographicAllocation /* properties: name */)

QUERY RULES:
- Use f.numberHoldings for count — do NOT count holdings manually.
- Weight on HAS_HOLDING is the portfolio weight.
- Weight on HAS_SECTOR_ALLOCATION / HAS_GEOGRAPHIC_ALLOCATION is allocation weight.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. STRICT SCHEMA ALIGNMENT: `payoffProfile`, `marketValue`, and `weight` are properties of the `[r:HAS_HOLDING]` relationship, NOT the `Holding` node. 
2. INLINE MATH: NEVER use math operators (`>`, `<`, `gt`) inside curly braces in MATCH patterns (e.g., BAD: `[r:HAS_HOLDING {marketValue: > 10000}]`). ALWAYS use a `WHERE` clause (e.g., `WHERE r.marketValue > 10000`).
3. DIVISION BY ZERO: When calculating percentage ratios, ALWAYS use `CASE WHEN total = 0 THEN 0 ELSE (part * 100.0 / total) END`.
4. COMMAS: Ensure proper commas are placed between variables in the `RETURN` statement before aggregate functions (e.g., `RETURN a, COUNT(b)`).
""",

    "fund_profile": """
Relevant schema (use vector similarity search on these nodes):
(:Fund /* properties: ticker, name */)-[:DEFINED_BY /* properties: date */]->(:Profile /* properties: id, summaryProspectus */)
(:Profile)-[:HAS_STRATEGY]->(:StrategyChunk /* properties: id, title, text, embedding */)
(:Profile)-[:HAS_RISK_NODE]->(:RiskChunk /* properties: id, title, text, embedding */)
(:Profile)-[:HAS_OBJECTIVE]->(:Objective /* properties: id, text, embedding */)
(:Profile)-[:HAS_PERFORMANCE_COMMENTARY]->(:PerformanceCommentary /* properties: id, text, embedding */)

QUERY RULES:
- These nodes have embeddings — use vector search for semantic queries.
- Vector indexes exist on embedding properties.
- ALWAYS return ticker and name alongside the requested data.
- Do NOT attempt to extract netAssets or numerical performance from these text nodes.
""",

    "company_filing": """
Relevant schema:
(:Company /* properties: ticker, name, cik */)-[:HAS_FILING /* properties: date */]->(filing:Filing10K /* properties: id */)
(filing)-[:HAS_RISK_FACTORS]->(:Section:RiskFactor /* properties: id, text, embedding */)
(filing)-[:HAS_BUSINESS_INFORMATION]->(:Section:BusinessInformation /* properties: id, text, embedding */)
(filing)-[:HAS_MANAGEMENT_DISCUSSION]->(:Section:ManagemetDiscussion /* properties: id, text, embedding */)
(filing)-[:HAS_LEGAL_PROCEEDINGS]->(:Section:LegalProceeding /* properties: id, text, embedding */)
(filing)-[:HAS_FINACIALS]->(fin:Section:Financials /* properties: incomeStatement, balanceSheet, cashFlow, fiscalYear */)
(fin)-[:HAS_METRIC]->(:FinancialMetric /* properties: label, value */)
(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment /* properties: label, value, percentage */)
(filing)-[:EXTRACTED_FROM]->(:Document /* properties: accessionNumber, url, filingDate */)

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
(:Company /* properties: ticker, name, cik */)-[:EMPLOYED_AS_CEO /* properties: ceoCompensation, ceoActuallyPaid, date */]->(:Person /* properties: name */)
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction /* properties: position, transactionType, shares, price, value, remainingShares */)-[:MADE_BY]->(:Person /* properties: name */)
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage /* properties: totalCompensation, shareholderReturn, date */)
(:CompensationPackage)-[:AWARDED_BY]->(:Company)

QUERY RULES:
- Fund managers are linked via MANAGED_BY.
- Company CEOs are linked via EMPLOYED_AS_CEO.
- Use personNameIndex for fuzzy person name search.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
2. INLINE MATH: NEVER use math operators (`>`, `<`) inside node patterns. Use `WHERE` clauses.
""",

    "hybrid_graph_vector": """
Relevant schema (combines graph traversal with vector similarity):

# Fund structure (for filtering)
(:Provider /* properties: name */)-[:MANAGES]->(:Trust /* properties: name */)-[:ISSUES]->(f:Fund /* properties: ticker, name, expenseRatio, netAssets, turnoverRate, advisoryFees, numberHoldings, costsPer10k */)

# Semantic content (for vector search)
(:Fund)-[:DEFINED_BY]->(:Profile /* properties: summaryProspectus */)
(:Profile)-[:HAS_STRATEGY]->(:StrategyChunk /* properties: id, title, text, embedding */)
(:Profile)-[:HAS_RISK_NODE]->(:RiskChunk /* properties: id, title, text, embedding */)
(:Profile)-[:HAS_OBJECTIVE]->(:Objective /* properties: id, text, embedding */)
(:Profile)-[:HAS_PERFORMANCE_COMMENTARY]->(:PerformanceCommentary /* properties: id, text, embedding */)

# Performance & Portfolio
(:Fund)-[:HAS_TRAILING_PERFORMANCE]->(:TrailingPerformance /* properties: return1y, return5y, return10y */)
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT /* properties: year */]->(:FinancialHighlight /* properties: turnover, expenseRatio, totalReturn */)
(:Fund)-[:HAS_PORTFOLIO]->(:Portfolio)-[:HAS_HOLDING /* properties: weight */]->(:Holding /* properties: name, ticker */)

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`.
2. DIVISION BY ZERO: When calculating percentage ratios, ALWAYS use `CASE WHEN total = 0 THEN 0 ELSE (part * 100.0 / total) END`.
""",

    "cross_entity": """
Relevant schema (multi-domain traversal):
(:Provider /* properties: name */)-[:MANAGES]->(:Trust /* properties: name */)-[:ISSUES]->(f:Fund /* properties: ticker, name */)
(:Fund)-[:MANAGED_BY /* properties: date */]->(:Person /* properties: name */)
(:Fund)-[:HAS_PORTFOLIO]->(:Portfolio)-[:HAS_HOLDING /* properties: weight, shares, marketValue */]->(h:Holding /* properties: name, ticker */)
(:Holding)-[:REPRESENTS]->(c:Company /* properties: ticker, name, cik */)
(:Company)-[:EMPLOYED_AS_CEO /* properties: ceoCompensation, ceoActuallyPaid */]->(:Person /* properties: name */)
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction /* properties: position, transactionType, shares, price, value, remainingShares */)-[:MADE_BY]->(:Person)

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. INLINE MATH: NEVER use math/comparison operators (`>`, `<`, `gt`) inside curly braces in MATCH patterns (e.g., BAD: `[r:HAS_HOLDING {marketValue: > 10000}]`). ALWAYS use a `WHERE` clause (e.g., `WHERE r.marketValue > 10000`).
2. PROPERTY SYNTAX: NEVER generate incomplete property filters like `(c:Company {ticker, name})`. Only use specific values like `{ticker: 'AAPL'}` or omit the brackets entirely if you are just defining variables.
3. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`.
"""
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

    # Categories that need both graph traversal + vector similarity search
    HYBRID_CATEGORIES = {
        QueryCategory.HYBRID_GRAPH_VECTOR.value,
        QueryCategory.FUND_PROFILE.value,
    }

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

        print("✓ QueryHandler ready")

    # ── public API ────────────────────────────────────────────────────────

    def handle(
        self,
        user_query: str,
        execute: bool = False,
        temperature: Optional[float] = None,
    ) -> QueryResult:
        """
        Full pipeline: classify → select schema → translate → optionally execute.

        Args:
            user_query:  Natural-language question.
            execute:     If True, run the generated Cypher against Neo4j
                         and attach the rows to ``result.data``.
            temperature: Override temperature for the Text2Cypher LLM.

        Returns:
            QueryResult with category, cypher, and optional data.
        """
        import time
        start_pipeline = time.time()
        
        # Step 1 — classify
        start_classification = time.time()
        prediction = self.classifier.predict(user_query)
        classification_time = time.time() - start_classification
        
        category = prediction["label"]
        confidence = prediction["confidence"]
        top_2 = prediction.get("top_2", [])

        if top_2 and len(top_2) == 2:
            cat1, conf1 = top_2[0]
            cat2, conf2 = top_2[1]
            logger.info(f"Classification: 1. {cat1} ({conf1:.2%}) | 2. {cat2} ({conf2:.2%})")
            print(f"🏷️  Top Matches: 1️⃣ {cat1} ({conf1:.2%}) | 2️⃣ {cat2} ({conf2:.2%})")
        else:
            cat1, conf1 = category, confidence
            cat2, conf2 = None, 0.0
            logger.info(f"Classification: {category} ({confidence:.2%})")
            print(f"🏷️  Category: {category}  (confidence {confidence:.2%})")

        # Step 2 — pick schema slice
        # Low-confidence fallback → widest schema
        if confidence < self.confidence_threshold:
            logger.warning(
                f"Low confidence ({confidence:.2%}), falling back to cross_entity"
            )
            print(f"⚠️  Low confidence — falling back to cross_entity schema")
            selected_schema_name = QueryCategory.CROSS_ENTITY.value
            schema_slice = SCHEMA_SLICES.get(selected_schema_name, "")
        elif top_2 and conf1 < 0.85 and cat2 != QueryCategory.NOT_RELATED.value:
            print(f"⚖️  Medium confidence -> Injecting combined schema for '{cat1}' and '{cat2}'")
            schema1 = SCHEMA_SLICES.get(cat1, "")
            schema2 = SCHEMA_SLICES.get(cat2, "")
            schema_slice = f"=== COMBINED SCHEMA OPTIONS ===\n\n--- Option A ({cat1}) ---\n{schema1}\n\n--- Option B ({cat2}) ---\n{schema2}"
            selected_schema_name = f"{cat1} + {cat2}"
        else:
            selected_schema_name = category
            schema_slice = SCHEMA_SLICES.get(category, "")

        # Step 3 — route
        result = QueryResult(
            query=user_query,
            category=category,
            confidence=confidence,
            schema_used=selected_schema_name,
        )

        # not_related → short-circuit
        if category == QueryCategory.NOT_RELATED.value:
            result.error = "Query is not related to the financial database."
            print("🚫 Not related — skipping.")
            return result

        # hybrid (includes fund_profile): flag vector AND generate Cypher
        if category in self.HYBRID_CATEGORIES or (selected_schema_name == f"{cat1} + {cat2}" and cat2 in self.HYBRID_CATEGORIES):
            result.requires_vector_search = True
            print("🔀 Hybrid — will generate Cypher AND flag vector search.")

        # Step 4 — translate with the sliced schema
        start_translation = time.time()
        cypher = self._translate_with_schema(
            user_query, schema_slice, temperature=temperature
        )
        translation_time = time.time() - start_translation
        result.cypher = cypher

        if cypher is None:
            result.error = "Failed to generate Cypher query."
            return result

        # Step 5 — optionally execute
        if execute and cypher:
            result.data = self._execute(cypher)

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
