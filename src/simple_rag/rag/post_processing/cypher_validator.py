"""
Cypher Query Validator Module.

Validates Cypher queries for:
1. READ-ONLY safety (no write operations allowed)
2. Syntax correctness (via Neo4j EXPLAIN)
3. Schema compliance (valid labels, properties, relationships)

SECURITY FEATURE: Write Operations Blocked
- By default, block_writes=True prevents all data modification
- Blocks: CREATE, MERGE, DELETE, SET, REMOVE, DROP, ALTER
- Only READ-ONLY queries allowed: MATCH, RETURN, WHERE, etc.
- Protects against accidental or malicious data changes

Uses cypher-guard for syntax checking and implements custom lightweight schema validation.

NOTE: cypher-guard's validate_cypher() hangs on certain queries and is NOT used.
NOTE: cypher-guard outputs verbose Rust DEBUG traces to stdout. When running
      benchmarks, redirect stdout or use `2>/dev/null | grep` to filter noise.
      This module itself does NOT attempt to suppress Rust output because
      OS-level fd redirection causes deadlocks with the Rust runtime.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a Cypher query validation."""
    is_valid: bool
    original_query: str
    syntax_errors: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)
    is_read_query: Optional[bool] = None
    is_write_query: Optional[bool] = None
    triggered_rule: Optional[str] = None  # short tag identifying which check fired first

    @property
    def all_errors(self) -> List[str]:
        return self.syntax_errors + self.schema_errors

    def __str__(self) -> str:
        if self.is_valid:
            return "✅ Query is valid"
        parts = ["❌ Query validation failed:"]
        for err in self.syntax_errors:
            parts.append(f"  [SYNTAX] {err}")
        for err in self.schema_errors:
            parts.append(f"  [SCHEMA] {err}")
        return "\n".join(parts)


class CypherValidator:
    """
    Validates Cypher queries for syntax correctness and schema compliance.
    
    - Syntax checking via cypher-guard (fast, reliable)
    - Read/write classification via cypher-guard
    - Schema validation via custom lightweight regex-based checker
    
    Usage:
        validator = CypherValidator()
        result = validator.validate("MATCH (f:Fund {ticker: 'VTI'}) RETURN f.name")
        if result.is_valid:
            ...  # safe to execute
        else:
            print(result)  # shows syntax/schema errors
    """

    # ─── Schema: valid node labels and their properties ──────────────────
    NODE_PROPERTIES: Dict[str, Set[str]] = {
        "Provider": {"name"},
        "Trust": {"name"},
        "Fund": {"ticker", "name", "cik", "securityExchange", "updatedAt"},
        "ShareClass": {"name", "description"},
        "Profile": {"summaryProspectus"},
        "Table": {"content", "title"},
        "Image": {"id", "title", "category", "svg"},
        "Sector": {"name"},
        "Region": {"name"},
        "AverageReturns": {"return1y", "return5y", "return10y", "returnInception"},
        "Person": {"name"},
        "Portfolio": {"id", "ticker", "count", "seriesId", "date"},
        "Holding": {"id", "name", "ticker", "cusip", "isin", "lei", "country",
                    "category", "category_desc", "issuerCategory", "issuer_category",
                    "issuerDesc", "businessAddress", "assetCategory", "assetDesc"},
        "FinancialHighlight": {"id", "turnover", "expenseRatio", "totalReturn",
                               "netAssets", "distributionShares",
                               "netAssetsValueBeginning", "netAssetsValueEnd",
                               "netIncomeRatio", "numberHoldings", "advisoryFees", "costsPer10k"},
        "Company": {"ticker", "name", "cik"},
        "CompensationPackage": {"totalCompensation", "shareholderReturn", "date"},
        "InsiderTransaction": {"transactionDate", "position", "transactionType", "shares",
                               "price", "value", "remainingShares"},
        "Chunk": {"title", "text", "embedding", "chunkType", "chunkIndex",
                  "subsection", "sectionType", "sectionName", "ticker", "filingDate",
                  "wordCount"},
        "Filing10K": set(),
        "Section": {"title", "text", "embedding", "sectionType", "secItem"},
        "Risk":       {"title", "text", "embedding"},
        "RiskFactor": {"title", "text", "embedding", "sectionType", "secItem"},
        "BusinessInformation": {"title", "text", "embedding", "sectionType", "secItem"},
        "LegalProceeding": {"title", "text", "embedding", "sectionType", "secItem"},
        "ManagementDiscussion": {"title", "text", "embedding", "sectionType", "secItem"},
        "Properties": {"title", "text", "embedding", "sectionType", "secItem"},
        "Objective": {"text", "title", "embedding"},
        "PerformanceCommentary": {"text", "title", "embedding"},
        "Strategy": {"text", "title", "embedding"},
        "Financials": {"incomeStatement", "balanceSheet", "cashFlow", "fiscalYear", "fiscalYeat"},
        "FinancialMetric": {"label", "value"},
        "Segment": {"label", "value", "percentage"},
        "Document": {"accession_number", "url", "form", "type",
                     "filing_date", "filingdate", "filingDate",
                     "reporting_date", "reportingDate",
                     "accessionNumber", "accesionNumber"},
        "AssetCategory": {"code", "name", "type", "subtype"},
    }

    # ─── Schema: valid relationship types and their (start, end) pairs ───
    RELATIONSHIPS: Dict[str, Set[Tuple[str, str]]] = {
        "MANAGES": {("Provider", "Trust")},
        "ISSUES": {("Trust", "Fund")},
        "HAS_SHARE_CLASS": {("Fund", "ShareClass")},
        "HAS_CHART": {("Fund", "Image")},
        "HAS_AVERAGE_RETURNS": {("Fund", "AverageReturns")},
        "HAS_SECTOR_ALLOCATION": {("Fund", "Sector")},
        "HAS_REGION_ALLOCATION": {("Fund", "Region")},
        "MANAGED_BY": {("Fund", "Person")},
        "HAS_PORTFOLIO": {("Fund", "Portfolio")},
        "HAS_FINANCIAL_HIGHLIGHT": {("Fund", "FinancialHighlight")},
        "DEFINED_BY": {("Fund", "Profile")},
        "HAS_HOLDING": {("Portfolio", "Holding")},
        "REPRESENTS": {("Holding", "Company")},
        "HAS_CEO": {("Company", "Person")},
        "AWARDED_BY": {("CompensationPackage", "Company")},
        "HAS_INSIDER_TRANSACTION": {("Company", "InsiderTransaction")},
        "MADE_BY": {("InsiderTransaction", "Person")},
        "RECEIVED_COMPENSATION": {("Person", "CompensationPackage")},
        "DISCLOSED_IN": {("CompensationPackage", "Document"),
                         ("InsiderTransaction", "Document")},
        "REPORTS_IN": {("Company", "Filing10K")},
        "HAS_TABLE": {("Fund", "Table")},
        "HAS_SECTION": {("Filing10K", "Section"), ("Filing10K", "RiskFactor"),
                        ("Filing10K", "BusinessInformation"), ("Filing10K", "LegalProceeding"),
                        ("Filing10K", "ManagementDiscussion"), ("Filing10K", "Properties"),
                        ("Profile", "Section"), ("Profile", "Objective"),
                        ("Profile", "PerformanceCommentary"), ("Profile", "Risk"),
                        ("Profile", "Strategy")},
        "HAS_FINANCIALS": {("Filing10K", "Financials")},
        "HAS_METRIC": {("Financials", "FinancialMetric")},
        "HAS_SEGMENT": {("FinancialMetric", "Segment")},
        "EXTRACTED_FROM": {("Fund", "Document"), ("Profile", "Document"),
                           ("Portfolio", "Document"), ("Filing10K", "Document"),
                           ("InsiderTransaction", "Document")},
        "HAS_CHUNK": {("Section", "Chunk"),
                      ("Risk", "Chunk"),
                      ("RiskFactor", "Chunk"),
                      ("BusinessInformation", "Chunk"),
                      ("LegalProceeding", "Chunk"),
                      ("ManagementDiscussion", "Chunk"),
                      ("Properties", "Chunk"),
                      ("Strategy", "Chunk"), ("Objective", "Chunk"),
                      ("PerformanceCommentary", "Chunk")},
        "OF_ASSET_TYPE": {("Holding", "AssetCategory")},
    }

    # Flat sets for quick lookup
    VALID_REL_TYPES: Set[str] = set(RELATIONSHIPS.keys())
    VALID_LABELS: Set[str] = set(NODE_PROPERTIES.keys())

    # Relationship properties
    REL_PROPERTIES: Dict[str, Set[str]] = {
        "DEFINED_BY": {"year"},
        "HAS_CHART": {"year"},
        "HAS_SECTOR_ALLOCATION": {"weight", "year"},
        "HAS_REGION_ALLOCATION": {"weight", "year"},
        "HAS_AVERAGE_RETURNS": {"year"},
        "MANAGED_BY": {"year"},
        "HAS_CEO": {"ceoCompensation", "ceoActuallyPaid", "date"},
        "HAS_HOLDING": {"shares", "marketValue", "weight", "currency",
                        "fairValueLevel", "isRestricted", "payoffProfile"},
        "HAS_FINANCIAL_HIGHLIGHT": {"year"},
        "REPORTS_IN": {"year"},
        "HAS_TABLE": {"year"},
        "HAS_PORTFOLIO": {"date"},
        "EXTRACTED_FROM": {"date"},
    }

    # Relationship direction auto-fixes.
    # Each entry: (wrong_regex, replacement, rel_name, reason)
    # Applied unconditionally before any validation step.
    # Only include relationships whose reversed form is NEVER semantically valid
    # (i.e. the LLM always writes them in the canonical direction but sometimes
    # flips the arrow accidentally).
    DIRECTION_FIXES: list = [
        # InsiderTransaction -[:MADE_BY]-> Person
        # LLM error: (it:InsiderTransaction)<-[:MADE_BY]-(p:Person)
        (r'<-\[:MADE_BY\]-(?!>)', '-[:MADE_BY]->', 'MADE_BY',
         'InsiderTransaction-[:MADE_BY]->Person (not reversed)'),
        # Company -[:HAS_INSIDER_TRANSACTION]-> InsiderTransaction
        # LLM error: (it:InsiderTransaction)<-[:HAS_INSIDER_TRANSACTION]-(c:Company)
        (r'<-\[:HAS_INSIDER_TRANSACTION\]-(?!>)', '-[:HAS_INSIDER_TRANSACTION]->', 'HAS_INSIDER_TRANSACTION',
         'Company-[:HAS_INSIDER_TRANSACTION]->InsiderTransaction (not reversed)'),
        # Filing10K -[:HAS_SECTION]-> Section
        (r'<-\[:HAS_SECTION\]-(?!>)', '-[:HAS_SECTION]->', 'HAS_SECTION',
         'Filing10K/Profile-[:HAS_SECTION]->Section (not reversed)'),
        # Filing10K -[:HAS_FINANCIALS]-> Financials
        (r'<-\[:HAS_FINANCIALS\]-(?!>)', '-[:HAS_FINANCIALS]->', 'HAS_FINANCIALS',
         'Filing10K-[:HAS_FINANCIALS]->Financials (not reversed)'),
        # Section/Chunk relationships
        (r'<-\[:HAS_CHUNK\]-(?!>)', '-[:HAS_CHUNK]->', 'HAS_CHUNK',
         'Section-[:HAS_CHUNK]->Chunk (not reversed)'),
        # Fund -[:HAS_FINANCIAL_HIGHLIGHT]-> FinancialHighlight
        (r'<-\[:HAS_FINANCIAL_HIGHLIGHT\]-(?!>)', '-[:HAS_FINANCIAL_HIGHLIGHT]->', 'HAS_FINANCIAL_HIGHLIGHT',
         'Fund-[:HAS_FINANCIAL_HIGHLIGHT]->FinancialHighlight (not reversed)'),
    ]

    # Write operations that are blocked
    BLOCKED_WRITE_KEYWORDS = [
        "CREATE ", "CREATE(",
        "MERGE ", "MERGE(",
        "DELETE ", "DETACH DELETE",
        "SET ",
        "REMOVE ",
        "DROP ",
        "ALTER ",
    ]

    def __init__(self, neo4j_driver=None, block_writes: bool = True, use_syntax_check: bool = True):
        """
        Initialize the CypherValidator.

        Args:
            neo4j_driver: Neo4j driver instance to run EXPLAIN queries for validation.
            block_writes: If True, mark write queries as invalid (DEFAULT: True).
                         This prevents all CREATE, DELETE, MERGE, SET, REMOVE operations.
            use_syntax_check: If True, use Neo4j EXPLAIN for syntax checking.
                             Set to False to skip syntax check (faster, less noise).

        Security Note:
            By default, block_writes=True ensures that only READ-only queries are allowed.
            This prevents any accidental or malicious data modification.
        """
        self.driver = neo4j_driver
        self.block_writes = block_writes
        self.use_syntax_check = use_syntax_check

    def _fix_avg_list_literal(self, query: str) -> tuple[str, list[str]]:
        """
        Rewrite avg([a, b, c]) → (a + b + c) / 3.0 before validation.
        Neo4j's avg() only accepts aggregation over a stream, not a list literal.
        """
        fixes = []
        def _replace(m):
            inner = m.group(1)
            items = [x.strip() for x in inner.split(',')]
            n = len(items)
            expr = ' + '.join(items)
            fixes.append(f"AVG_LIST_FIX: avg([{inner}]) → ({expr}) / {n}.0")
            return f"({expr}) / {n}.0"
        new_query = re.sub(r'\bavg\s*\(\s*\[([^\]]+)\]\s*\)', _replace, query, flags=re.IGNORECASE)
        return new_query, fixes

    # Properties stored as native floats in Neo4j — toFloat() is a no-op but
    # prevents the query planner from using range indexes for ORDER BY / WHERE.
    _NATIVE_FLOAT_PROPS = {
        "r.weight", "r.marketValue", "r.shares",
        "fh.expenseRatio", "fh.totalReturn", "fh.netAssets",
        "fh.turnover", "fh.advisoryFees", "fh.netIncomeRatio",
        "fh.netAssetsValueBeginning", "fh.netAssetsValueEnd",
        "ar.return1y", "ar.return5y", "ar.return10y", "ar.returnInception",
        "ra.weight", "sa.weight", "h.weight",
    }

    def _fix_redundant_tofloat(self, query: str) -> tuple[str, list[str]]:
        """Remove toFloat() wrapping on properties already stored as floats.

        toFloat() on an already-float property is a no-op semantically but it
        prevents Neo4j's query planner from using range indexes for ORDER BY
        and WHERE comparisons, causing full relationship scans on large portfolios.
        """
        fixes: list[str] = []
        for prop in self._NATIVE_FLOAT_PROPS:
            pattern = rf'\btoFloat\s*\(\s*({re.escape(prop)})\s*\)'
            def _repl(m: re.Match, p: str = prop) -> str:
                fixes.append(f"TOFLOAT_FIX: removed redundant toFloat({p})")
                return p
            query = re.sub(pattern, _repl, query, flags=re.IGNORECASE)
        return query, fixes

    def _fix_relationship_directions(self, query: str) -> tuple[str, list[str]]:
        """
        Auto-correct known relationship direction mistakes before validation.

        Returns (fixed_query, list_of_fixes_applied).
        Only rewrites relationships whose reversed form is never semantically
        valid — so the fix is always safe to apply unconditionally.
        """
        fixes_applied = []
        for pattern, replacement, rel_name, reason in self.DIRECTION_FIXES:
            new_query, n = re.subn(pattern, replacement, query)
            if n:
                fixes_applied.append(f"DIRECTION_FIX [{rel_name}]: {reason} — fixed {n} occurrence(s)")
                query = new_query
        return query, fixes_applied

    def validate(self, query: str) -> ValidationResult:
        """
        Validate a Cypher query for syntax and schema compliance.
        """
        result = ValidationResult(is_valid=True, original_query=query)

        if not query or not query.strip():
            result.is_valid = False
            result.syntax_errors.append("Empty query")
            return result

        query = query.strip()

        # Strip markdown fence
        if query.startswith("```"):
            lines = query.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            query = "\n".join(lines).strip()

        # Auto-fix common relationship direction mistakes before any other check
        query, direction_fixes = self._fix_relationship_directions(query)
        for fix_msg in direction_fixes:
            logger.info(fix_msg)
            print(f"🔧 {fix_msg}")

        # Auto-fix avg([list]) → arithmetic mean expression
        query, avg_fixes = self._fix_avg_list_literal(query)
        for fix_msg in avg_fixes:
            logger.info(fix_msg)
            print(f"🔧 {fix_msg}")

        # Auto-fix redundant toFloat() on properties already stored as floats
        query, tofloat_fixes = self._fix_redundant_tofloat(query)
        for fix_msg in tofloat_fixes:
            logger.info(fix_msg)
            print(f"🔧 {fix_msg}")

        # Inject Provider and Trust for single-fund queries if not already present
        query, pt_modified = CypherValidator.inject_fund_provider_trust(query)
        if pt_modified:
            logger.info("PROVIDER_TRUST_INJECT: added provider/trust context to fund query")
            print("🔧 PROVIDER_TRUST_INJECT: provider and trust added to RETURN")

        # Step 0: Catch EXPLAIN / PROFILE prefix — the model sometimes adds these debug keywords
        if re.match(r'^(EXPLAIN|PROFILE)\s+', query, re.IGNORECASE):
            result.is_valid = False
            result.triggered_rule = "EXPLAIN_PREFIX"
            result.syntax_errors.append(
                "EXPLAIN/PROFILE PREFIX ERROR: Your query starts with EXPLAIN or PROFILE. "
                "These are Neo4j debug keywords, not valid query prefixes. "
                "Return ONLY the Cypher query without any prefix. "
                "BAD:  EXPLAIN MATCH (f:Fund {ticker: 'VTI'}) RETURN f.name "
                "GOOD: MATCH (f:Fund {ticker: 'VTI'}) RETURN f.name"
            )
            return result

        # Step 0a-extra: Catch variable collision — same name used for rel and node
        # e.g. MATCH (f:Fund)-[p:HAS_PORTFOLIO]->(p:Portfolio) — fatal Cypher error
        _rel_vars = set(re.findall(r'\[\s*([a-zA-Z_]\w*)\s*:[A-Z_]', query))
        _node_vars = set(re.findall(r'\(\s*([a-zA-Z_]\w*)\s*:[A-Za-z_]', query))
        _collisions = _rel_vars & _node_vars
        if _collisions:
            _collision_var = next(iter(_collisions))
            result.is_valid = False
            result.triggered_rule = "VARIABLE_COLLISION"
            result.syntax_errors.append(
                f"VARIABLE COLLISION ERROR: The variable '{_collision_var}' is used for both a "
                f"relationship and a node in the same query. This is invalid Cypher and will crash. "
                f"Use distinct variable names"
                f"\nBAD:  MATCH (f:Fund)-[p:HAS_PORTFOLIO]->(p:Portfolio) "
                f"\nGOOD: MATCH (f:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio)"
            )
            return result

        # Step 0a-extra2: Catch invalid Neo4j function calls: year(), month(), day()
        # Neo4j does not have these as standalone functions.
        # To extract a date part, use field accessor: date(prop).year
        _invalid_fn = re.search(r'\b(year|month|day)\b\s*\(', query, re.IGNORECASE)
        if _invalid_fn:
            _fn_name = _invalid_fn.group(1).lower()
            result.is_valid = False
            result.triggered_rule = "INVALID_DATE_FUNCTION_CALL"
            result.syntax_errors.append(
                f"INVALID FUNCTION ERROR: Neo4j has no '{_fn_name}()' function. "
                f"To extract a date part from a date property, use field accessor syntax: "
                f"date(prop).year, date(prop).month, date(prop).day. "
                f"BAD:  WHERE year(it.transactionDate) = 2023 "
                f"GOOD: WHERE date(it.transactionDate).year = 2023"
            )
            return result

        # Step 0a-extra3: Catch contains() used as a function (it is a keyword operator, not a function)
        _contains_fn = re.search(r'\bcontains\s*\(', query, re.IGNORECASE)
        if _contains_fn:
            result.is_valid = False
            result.triggered_rule = "INVALID_CONTAINS_FUNCTION"
            result.syntax_errors.append(
                "INVALID FUNCTION ERROR: 'contains()' is not a valid Neo4j function. "
                "Use the CONTAINS keyword as a string operator instead. "
                "BAD:  WHERE contains(f.name, 'Vanguard') "
                "GOOD: WHERE f.name CONTAINS 'Vanguard'"
            )
            return result

        # Step 0a-extra: Catch non-existent Neo4j date/year functions
        # e.g. years(currentDate()), YEAR(DATE()), CURRENT_DATE, year(datetime())
        _invalid_date = re.search(
            r'\b(years|months|days)\s*\(\s*(currentDate|datetime|date)\s*\(\)'
            r'|CURRENT_DATE\b'
            r'|\bYEAR\s*\(\s*(DATE|CURRENT)',
            query, re.IGNORECASE
        )
        if _invalid_date:
            result.is_valid = False
            result.triggered_rule = "INVALID_DATE_FUNCTION"
            result.syntax_errors.append(
                "INVALID DATE FUNCTION: Neo4j does not support years(), YEAR(), CURRENT_DATE, "
                "or similar date arithmetic functions for computing the current year. "
                "Do NOT compute the current year dynamically. Instead, return all rows ordered "
                "by year DESC and use LIMIT or list indexing to select the relevant period. "
                "BAD:  WHERE r.year >= years(currentDate()) - 2 "
                "BAD:  WHERE r.year = YEAR(DATE()) - 1 "
                "GOOD: MATCH (f:Fund {ticker: 'VIEIX'})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) "
                "      WHERE fh.netAssetsValueBeginning IS NOT NULL "
                "      RETURN f.ticker AS ticker, r.year AS year, fh.netAssetsValueBeginning AS netAssetValue "
                "      ORDER BY r.year DESC"
            )
            return result

        # Step 0-fh: Catch FinancialHighlight properties accessed on Fund variable
        # e.g. f.expenseRatio, f.netAssets, f.advisoryFees — these live on FinancialHighlight, not Fund
        _fh_only_props = {'expenseRatio', 'netAssets', 'advisoryFees', 'totalReturn',
                          'turnover', 'netIncomeRatio', 'netAssetsValueBeginning',
                          'netAssetsValueEnd', 'costsPer10k'}
        # Find Fund variables: (f:Fund) or (fund:Fund)
        _fund_vars = set(re.findall(r'\(\s*(\w+)\s*:\s*Fund\b', query))
        if _fund_vars:
            for fvar in _fund_vars:
                for prop in _fh_only_props:
                    if re.search(rf'\b{re.escape(fvar)}\.{re.escape(prop)}\b', query):
                        result.is_valid = False
                        result.triggered_rule = "FH_PROPERTY_ON_FUND"
                        result.syntax_errors.append(
                            f"WRONG NODE PROPERTY ERROR: '{fvar}.{prop}' accesses '{prop}' on the Fund node, "
                            f"but '{prop}' is a property of FinancialHighlight, not Fund. "
                            f"You must traverse the HAS_FINANCIAL_HIGHLIGHT relationship first. "
                            f"BAD:  MATCH (f:Fund {{ticker: 'VTI'}}) RETURN f.{prop} "
                            f"GOOD: MATCH (f:Fund {{ticker: 'VTI'}})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) "
                            f"RETURN fh.{prop}, r.year ORDER BY r.year DESC LIMIT 1"
                        )
                        return result

        # Step 0: Fast pre-check — catch inline math operators inside {} before Neo4j
        inline_math = re.search(r'\{[^}]*(?:[<>]=?|!=)\s*[\d\w\'"]', query)
        if inline_math:
            result.is_valid = False
            result.triggered_rule = "INLINE_MATH"
            result.syntax_errors.append(
                "INLINE MATH ERROR: You used a comparison operator (>, <, >=, <=, !=) "
                "inside curly braces in a MATCH pattern. This is invalid Cypher syntax. "
                "Move the condition to a WHERE clause after MATCH. "
                "BAD:  MATCH (n:Node {prop: > 10}) "
                "GOOD: MATCH (n:Node)-[r:REL]->(m) WHERE r.prop > 10"
            )
            return result

        # Step 0a: Catch IS NOT NULL, IS NULL, and other operators inside {} property filters
        # These are conditional operators, not values, and cannot be used in property maps
        null_check_in_braces = re.search(
            r'\{[^}]*:\s*(?:IS\s+)?(?:NOT\s+)?NULL\s*[,}]',
            query,
            re.IGNORECASE
        )
        if null_check_in_braces:
            result.is_valid = False
            result.triggered_rule = "NULL_IN_BRACES"
            result.syntax_errors.append(
                "NULL CHECK IN BRACES ERROR: You used IS NOT NULL, IS NULL, or NOT NULL "
                "inside curly braces {}. Property maps only accept exact values (strings, numbers, booleans), "
                "not conditional operators. Move NULL checks to a WHERE clause. "
                "BAD:  MATCH (ar:AverageReturns {returnInception: IS NOT NULL}) "
                "GOOD: MATCH (ar:AverageReturns) WHERE ar.returnInception IS NOT NULL"
            )
            return result
        
        # Step 0b: Catch other operators that don't belong in property maps
        # CONTAINS, STARTS WITH, ENDS WITH, IN, etc.
        operator_in_braces = re.search(
            r'\{[^}]*:\s*(?:CONTAINS|STARTS\s+WITH|ENDS\s+WITH|IN\s+\[)',
            query,
            re.IGNORECASE
        )
        if operator_in_braces:
            result.is_valid = False
            result.triggered_rule = "OPERATOR_IN_BRACES"
            result.syntax_errors.append(
                "OPERATOR IN BRACES ERROR: You used a conditional operator (CONTAINS, STARTS WITH, ENDS WITH, IN) "
                "inside curly braces {}. Property maps only accept exact values for equality matching. "
                "Move these operators to a WHERE clause. "
                "BAD:  MATCH (f:Fund {name: CONTAINS 'Vanguard'}) "
                "GOOD: MATCH (f:Fund) WHERE f.name CONTAINS 'Vanguard'"
            )
            return result

        # Step 0b: Catch WHERE clause placed after RETURN
        where_after_return = re.search(r'\bRETURN\b.+\bWHERE\b', query, re.IGNORECASE | re.DOTALL)
        if where_after_return:
            result.is_valid = False
            result.triggered_rule = "WHERE_AFTER_RETURN"
            result.syntax_errors.append(
                "CLAUSE ORDER ERROR: You placed a WHERE clause after RETURN. "
                "In Cypher, WHERE must come before RETURN (right after MATCH/WITH). "
                "BAD:  MATCH (f:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh) RETURN f.name WHERE fh.turnover > 10 "
                "GOOD: MATCH (f:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh) WHERE fh.turnover > 10 RETURN f.name"
            )
            return result

        # Step 0c: Catch aggregate functions used inside WHERE clauses (invalid Cypher)
        # e.g. WHERE year = max(r.year) - 2  →  aggregates cannot appear in predicates
        # Strip {} blocks first to avoid false-positives from EXISTS{} subqueries and literal maps
        _query_no_braces = re.sub(r'\{[^{}]*\}', '', query)
        where_agg = re.search(
            r'\bWHERE\b(?:(?!\b(?:RETURN|WITH|OPTIONAL\s+MATCH|MATCH|ORDER\s+BY|LIMIT|SKIP|UNION)\b).)*?\b(COUNT|SUM|AVG|MIN|MAX)\s*\(',
            _query_no_braces, re.IGNORECASE
        )
        if where_agg:
            agg_fn = where_agg.group(1).upper()
            result.is_valid = False
            result.triggered_rule = "AGGREGATE_IN_WHERE"
            result.syntax_errors.append(
                f"AGGREGATE IN WHERE ERROR: You used {agg_fn}() inside a WHERE clause. "
                "Aggregate functions (COUNT, SUM, AVG, MIN, MAX) cannot be used in WHERE predicates. "
                "To filter or compare against an aggregated value, compute it first in a WITH clause. "
                "BAD:  WITH f, r.year AS year WHERE year = max(r.year) - 2 "
                "GOOD: WITH f, r.year AS year ORDER BY year DESC "
                "      WITH f, collect(year) AS years "
                "      MATCH (f)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh) WHERE r.year = years[2]"
            )
            return result

        # Step 0d: Catch count(var)-[:REL]-> pattern — aggregate applied as if it were a node
        count_then_rel = re.search(r'\bcount\s*\([^)]*\)\s*-\s*\[', query, re.IGNORECASE)
        if count_then_rel:
            result.is_valid = False
            result.triggered_rule = "AGGREGATE_TRAVERSAL"
            result.syntax_errors.append(
                "AGGREGATE TRAVERSAL ERROR: You used count(...)-[:REL]->... which is invalid. "
                "count() returns a number, not a node — you cannot traverse a relationship from it. "
                "Fix: use OPTIONAL MATCH to traverse the path first, then aggregate in RETURN. "
                "BAD:  RETURN f.name, count(f)-[:HAS_PORTFOLIO]->(:Portfolio)-[:HAS_HOLDING]->(h:Holding) AS holdingsCount "
                "GOOD: OPTIONAL MATCH (f)-[:HAS_PORTFOLIO]->(:Portfolio)-[:HAS_HOLDING]->(h:Holding) "
                "      RETURN f.name, count(h) AS holdingsCount"
            )
            return result

        # Step 0e: Catch CALL db.index.fulltext/vector.queryNodes that uses 'score' without yielding it
        _fulltext_call = re.search(
            r'\bCALL\s+db\.index\.(fulltext|vector)\.queryNodes\b',
            query, re.IGNORECASE
        )
        if _fulltext_call and re.search(r'\bscore\b', query, re.IGNORECASE):
            _yield_m = re.search(
                r'\bYIELD\b\s+(.+?)(?=\s+(?:ORDER\s+BY|WHERE|LIMIT\b|MATCH\b|WITH\b|RETURN\b)|$)',
                query, re.IGNORECASE
            )
            _score_in_yield = bool(_yield_m and re.search(r'\bscore\b', _yield_m.group(1), re.IGNORECASE))
            if not _score_in_yield:
                result.is_valid = False
                result.triggered_rule = "FULLTEXT_SCORE_NOT_YIELDED"
                result.syntax_errors.append(
                    "FULLTEXT SCORE NOT YIELDED ERROR: You referenced 'score' (e.g. ORDER BY score DESC) "
                    "but did not include it in the YIELD clause of your CALL db.index.fulltext.queryNodes() call. "
                    "Every variable used after YIELD must be explicitly listed there. "
                    "BAD:  CALL db.index.fulltext.queryNodes('fundNameIndex', 'Vanguard~') YIELD node ORDER BY score DESC LIMIT 1 "
                    "MATCH (node)-[:DEFINED_BY]->(p:Profile) RETURN node.name, p.summaryProspectus "
                    "GOOD: CALL db.index.fulltext.queryNodes('fundNameIndex', 'Vanguard~') YIELD node, score ORDER BY score DESC LIMIT 1 "
                    "MATCH (node)-[:DEFINED_BY]->(p:Profile) RETURN node.name, p.summaryProspectus"
                )
                return result

        # Step 0f: Catch undefined relationship variables
        # Find all relationship variable usages (e.g., r.weight, r.year)
        undefined_rel_vars = self._check_undefined_relationship_variables(query)
        if undefined_rel_vars:
            result.is_valid = False
            result.triggered_rule = "UNDEFINED_REL_VAR"
            for var_name, property_names in undefined_rel_vars.items():
                result.syntax_errors.append(
                    f"UNDEFINED RELATIONSHIP VARIABLE ERROR: You used '{var_name}.{property_names[0]}' "
                    f"but the relationship variable '{var_name}' is not defined in your MATCH pattern.\n"
                    f"   In Cypher, you must capture relationship variables in square brackets: [r:REL_TYPE]\n"
                    f"   BAD:  MATCH (f:Fund)-[:HAS_REGION_ALLOCATION]->(g:Region) "
                    f"RETURN g.name, r.weight\n"
                    f"   GOOD: MATCH (f:Fund)-[r:HAS_REGION_ALLOCATION]->(g:Region) "
                    f"RETURN g.name AS region, r.weight AS weight\n"
                    f"   The variable '{var_name}' is referenced in your RETURN/WHERE/ORDER BY clause, "
                    f"so you MUST add it to the relationship pattern as [{var_name}:REL_TYPE]."
                )
            return result

        # Step 1: Write detection (keyword-based, no cypher-guard needed)
        self._classify_query(query, result)
        if self.block_writes and result.is_write_query:
            result.is_valid = False
            result.triggered_rule = "WRITE_OPERATION"
            # Extract which write operation was detected
            write_op = None
            upper = query.upper()
            for kw in self.BLOCKED_WRITE_KEYWORDS:
                if kw.strip() in upper:
                    write_op = kw.strip()
                    break

            result.syntax_errors.append(
                f"❌ WRITE OPERATION BLOCKED: {write_op or 'WRITE'}\n"
                f"   This system only allows READ-ONLY Cypher queries (MATCH, RETURN, WHERE, etc.)\n"
                f"   Write operations are disabled to protect the database:\n"
                f"   - CREATE: Cannot create new nodes or relationships\n"
                f"   - DELETE: Cannot delete nodes or relationships\n"
                f"   - MERGE: Cannot create or update data\n"
                f"   - SET: Cannot modify properties\n"
                f"   - REMOVE: Cannot remove properties\n"
                f"   - DROP: Cannot drop indexes or constraints\n"
                f"   Please reformulate your query using only MATCH, WHERE, RETURN, and other read-only clauses."
            )
            return result

        # Step 2: Syntax check via cypher-guard (optional — produces noisy output)
        if self.use_syntax_check:
            if not self._check_syntax(query, result):
                result.is_valid = False
                return result

        # Step 3: Lightweight schema validation
        self._validate_schema_lightweight(query, result)
        if result.schema_errors:
            result.is_valid = False
            unknown_labels = [e for e in result.schema_errors if e.startswith("Unknown node label")]
            unknown_rels   = [e for e in result.schema_errors if e.startswith("Unknown relationship type")]
            if unknown_labels and unknown_rels:
                result.triggered_rule = "UNKNOWN_LABEL+UNKNOWN_REL_TYPE"
            elif unknown_labels:
                result.triggered_rule = "UNKNOWN_LABEL"
            else:
                result.triggered_rule = "UNKNOWN_REL_TYPE"

        return result

    @staticmethod
    def replace_fund_name_with_resolved_ticker(
        cypher: str,
        resolved_entities: dict | None = None,
    ) -> tuple:
        """
        Replace (f:Fund {name: 'X'}) inline exact-name filters with the ticker
        identified by the entity resolver, when a close name match exists.

        Fund names in user queries are often partial or slightly wrong (missing
        "Index", "ETF", etc.), so exact name filters return 0 results.  When
        the resolver has already found a ticker for a similar name, using the
        ticker is always more reliable.

        Requires resolved_entities to have entries with type='Fund' and a 'ticker'.
        Uses word-overlap similarity (>= 70%) to identify matching resolved names.

        Returns (cleaned_cypher: str, was_modified: bool).
        """
        if resolved_entities is None:
            return cypher, False

        # Build a map: fund_name (lowercase) → ticker, from resolver results
        fund_name_to_ticker: dict = {}
        for entity_name, entity_info in resolved_entities.items():
            if isinstance(entity_info, dict):
                if entity_info.get("type") == "Fund" and entity_info.get("ticker"):
                    fund_name_to_ticker[entity_name.lower()] = entity_info["ticker"]

        if not fund_name_to_ticker:
            return cypher, False

        pattern = re.compile(
            r'(\(\s*\w*\s*:Fund\s*)\{\s*name\s*:\s*[\'"]([^\'"]+)[\'"]\s*\}(\s*\))',
            re.IGNORECASE,
        )

        # Also build a map that includes entity resolver score for tiebreaking
        fund_name_to_info: dict = {}
        for entity_name, entity_info in resolved_entities.items():
            if isinstance(entity_info, dict):
                if entity_info.get("type") == "Fund" and entity_info.get("ticker"):
                    fund_name_to_info[entity_name.lower()] = entity_info

        was_modified = [False]
        _STOP = {"fund", "vanguard", "ishares", "blackrock", "the", "of", "a", "index", "etf", "admiral", "shares"}

        def _replace(m):
            node_open = m.group(1)
            query_name = m.group(2)
            node_close = m.group(3)

            query_words = set(re.sub(r"[^a-z0-9 ]", "", query_name.lower()).split()) - _STOP
            if len(query_words) < 1:
                return m.group(0)

            best_ticker: str | None = None
            best_coverage = 0.0
            best_resolver_score = 0.0
            for resolved_name, info in fund_name_to_info.items():
                resolved_words = set(re.sub(r"[^a-z0-9 ]", "", resolved_name).split()) - _STOP
                if not resolved_words:
                    continue
                # "coverage" = what fraction of the query's meaningful words appear in the entity name
                coverage = len(query_words & resolved_words) / len(query_words)
                resolver_score = info.get("score", 0.0)
                if coverage > best_coverage or (coverage == best_coverage and resolver_score > best_resolver_score):
                    best_coverage = coverage
                    best_resolver_score = resolver_score
                    best_ticker = info["ticker"]

            if best_ticker and best_coverage >= 0.85:
                was_modified[0] = True
                return f"{node_open}{{ticker: '{best_ticker}'}}{node_close}"
            return m.group(0)

        cleaned = pattern.sub(_replace, cypher)
        return cleaned, was_modified[0]

    @staticmethod
    def strip_filing10k_year_filter(cypher: str) -> tuple:
        """
        Remove year property filters from Filing10K node patterns.

        Filing10K has NO year property — the year lives exclusively on the
        REPORTS_IN relationship (r.year).  When the model writes
        (f:Filing10K {year: 2025}) it always returns 0 results.

        Handles both single-property and mixed maps:
          (f:Filing10K {year: 2025})             → (f:Filing10K)
          (f:Filing10K {year: 2025, other: 'x'}) → (f:Filing10K {other: 'x'})

        Returns (cleaned_cypher: str, was_modified: bool).
        """
        if not re.search(r':Filing10K', cypher, re.IGNORECASE):
            return cypher, False

        was_modified = [False]

        def _strip_year(m):
            node_open = m.group(1)   # e.g. "(f:Filing10K "
            props_str = m.group(2)   # e.g. "year: 2025" or "year: 2025, other: 'x'"
            node_close = m.group(3)  # e.g. ")"

            # Remove "year: VALUE," or ", year: VALUE" or just "year: VALUE"
            cleaned = re.sub(
                r'\byear\s*:\s*(?:\d{4}|\$\w+)\s*(?:,\s*)?',
                '',
                props_str,
                flags=re.IGNORECASE,
            ).strip().strip(',').strip()

            was_modified[0] = True
            if cleaned:
                return f'{node_open}{{{cleaned}}}{node_close}'
            return node_open.rstrip() + node_close

        pattern = re.compile(
            r'(\(\s*\w*\s*:Filing10K\s*)\{([^}]*\byear\s*:\s*(?:\d{4}|\$\w+)[^}]*)\}(\s*\))',
            re.IGNORECASE,
        )
        cleaned = pattern.sub(_strip_year, cypher)
        return cleaned, was_modified[0]

    @staticmethod
    def strip_has_average_returns_year_filter(cypher: str) -> tuple:
        """
        Remove year property filters from HAS_AVERAGE_RETURNS relationship patterns.

        AverageReturns properties (return1y, return5y, return10y, returnInception)
        already encode the time period. A year filter on the relationship like
        [:HAS_AVERAGE_RETURNS {year: 2023}] returns 0 results when the stored year
        doesn't match, and is always wrong when the user asks about "best/highest
        returns" (they want all years, not just 2023).

        Handles:
          [:HAS_AVERAGE_RETURNS {year: 2023}]           → [:HAS_AVERAGE_RETURNS]
          [r:HAS_AVERAGE_RETURNS {year: 2023}]          → [r:HAS_AVERAGE_RETURNS]
          [:HAS_AVERAGE_RETURNS {year: 2023, other: 1}] → [:HAS_AVERAGE_RETURNS {other: 1}]

        Returns (cleaned_cypher: str, was_modified: bool).
        """
        if not re.search(r'HAS_AVERAGE_RETURNS', cypher, re.IGNORECASE):
            return cypher, False

        was_modified = [False]

        def _strip_year(m):
            rel_open = m.group(1)   # e.g. "[:HAS_AVERAGE_RETURNS " or "[r:HAS_AVERAGE_RETURNS "
            props_str = m.group(2)  # e.g. "year: 2023" or "year: 2023, other: 1"
            rel_close = m.group(3)  # e.g. "]"

            cleaned = re.sub(
                r'\byear\s*:\s*(?:\d{4}|\$\w+)\s*(?:,\s*)?',
                '',
                props_str,
                flags=re.IGNORECASE,
            ).strip().strip(',').strip()

            was_modified[0] = True
            if cleaned:
                return f'{rel_open}{{{cleaned}}}{rel_close}'
            return rel_open.rstrip() + rel_close

        pattern = re.compile(
            r'(\[\s*\w*\s*:HAS_AVERAGE_RETURNS\s*)\{([^}]*\byear\s*:\s*(?:\d{4}|\$\w+)[^}]*)\}(\s*\])',
            re.IGNORECASE,
        )
        cleaned = pattern.sub(_strip_year, cypher)

        # Also strip year filters placed on the AverageReturns NODE:
        # (ar:AverageReturns {year: 2023}) → (ar:AverageReturns)
        # AverageReturns has no year property — the LLM commonly makes this mistake.
        # Only apply to AverageReturns node contexts
        node_ctx_pattern = re.compile(
            r'(:AverageReturns\s*)\{([^}]*\byear\s*:\s*(?:\d{4}|\$\w+)[^}]*)\}',
            re.IGNORECASE,
        )

        def _strip_node_year_ctx(m):
            label = m.group(1)
            props_str = m.group(2)
            remaining = re.sub(
                r'\byear\s*:\s*(?:\d{4}|\$\w+)\s*(?:,\s*)?',
                '',
                props_str,
                flags=re.IGNORECASE,
            ).strip().strip(',').strip()
            was_modified[0] = True
            return label + ('{' + remaining + '}' if remaining else '')

        cleaned = node_ctx_pattern.sub(_strip_node_year_ctx, cleaned)
        return cleaned, was_modified[0]

    @staticmethod
    def strip_portfolio_intermediary_for_allocation(cypher: str) -> tuple:
        """
        Remove the spurious [:HAS_PORTFOLIO]->(p:Portfolio) hop when the model
        incorrectly routes HAS_SECTOR_ALLOCATION or HAS_REGION_ALLOCATION through
        Portfolio instead of directly from Fund.

        Schema: (Fund)-[HAS_SECTOR_ALLOCATION]->(Sector)  — direct, never via Portfolio.
        Model error: (Fund)-[:HAS_PORTFOLIO]->(p:Portfolio)-[sa:HAS_SECTOR_ALLOCATION]->(Sector)

        Input:  (f:Fund {ticker: 'VTI'})-[:HAS_PORTFOLIO]->(p:Portfolio)-[sa:HAS_SECTOR_ALLOCATION]->(s:Sector)
        Output: (f:Fund {ticker: 'VTI'})-[sa:HAS_SECTOR_ALLOCATION]->(s:Sector)

        Returns (cleaned_cypher: str, was_modified: bool).
        """
        if not re.search(r'HAS_PORTFOLIO', cypher, re.IGNORECASE):
            return cypher, False
        if not re.search(r'HAS_SECTOR_ALLOCATION|HAS_REGION_ALLOCATION', cypher, re.IGNORECASE):
            return cypher, False

        # Remove "-[:HAS_PORTFOLIO]->(var:Portfolio)" — the intermediate hop
        pattern = re.compile(
            r'-\[:HAS_PORTFOLIO\]->\s*\(\s*\w*\s*:Portfolio\s*\)',
            re.IGNORECASE,
        )
        cleaned, n = pattern.subn('', cypher)
        return cleaned, n > 0

    @staticmethod
    def strip_vector_where_filters(cypher: str) -> tuple:
        """
        In vector search queries, remove WHERE predicates that filter by year or
        text CONTAINS — these silently return 0 results because the post-YIELD MATCH
        cannot pre-filter the vector index.

        Kept: Company/Fund ticker/name inline filters ({ticker: 'XYZ'}).
        Removed: r.year = N, r.year IS NOT NULL, x.prop CONTAINS '...'.

        Returns (cleaned_cypher: str, was_modified: bool).
        """
        if 'db.index.vector.queryNodes' not in cypher:
            return cypher, False

        _BAD = [
            # year equality / range on relationship
            re.compile(r'^\s*r\.year\s*(?:=|!=|<>|<=?|>=?)\s*\d{4}\s*$', re.IGNORECASE),
            re.compile(r'^\s*r\.year\s+IS\s+NOT\s+NULL\s*$', re.IGNORECASE),
            # CONTAINS text filter on any node property (section, chunk, filing)
            re.compile(r'^\s*\w+\.\w+\s+(?:NOT\s+)?CONTAINS\s+[\'"][^\'"]*[\'"]\s*$', re.IGNORECASE),
        ]

        was_modified = [False]

        def _clean_where(m):
            content = m.group(1)
            parts = re.split(r'\s+AND\s+', content, flags=re.IGNORECASE)
            kept = []
            for p in parts:
                if any(pat.match(p) for pat in _BAD):
                    was_modified[0] = True
                else:
                    kept.append(p.strip())
            # Always append a space: the regex consumes the trailing whitespace
            # before CALL/MATCH/etc., so omitting it merges tokens (e.g. "NULLCALL").
            if not kept:
                return ' '
            return 'WHERE ' + ' AND '.join(kept) + ' '

        # Match WHERE clause up to the next major Cypher keyword
        cleaned = re.sub(
            r'\bWHERE\b\s+((?:(?!\b(?:RETURN|WITH|ORDER|LIMIT|MATCH|CALL|UNION)\b).)+)',
            _clean_where,
            cypher,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Collapse any double-spaces left by WHERE removal
        cleaned = re.sub(r'  +', ' ', cleaned).strip()
        return cleaned, was_modified[0]

    @staticmethod
    def inject_chunk_id(cypher: str) -> tuple:
        """
        Ensure vector search queries always include `chunk.id AS chunkId` in RETURN.
        Without it the benchmark (and result post-processing) cannot identify chunks.
        Inserts `chunk.id AS chunkId, ` right after the RETURN keyword when missing.
        Returns (cypher: str, was_modified: bool).
        """
        if 'db.index.vector.queryNodes' not in cypher:
            return cypher, False
        # Already present — nothing to do
        if re.search(r'chunk\.id\s+AS\s+chunkId', cypher, re.IGNORECASE):
            return cypher, False

        # Find the RETURN keyword and insert after it
        def _inject(m):
            return m.group(0) + 'chunk.id AS chunkId, '

        cleaned = re.sub(r'\bRETURN\s+(?:DISTINCT\s+)?', _inject, cypher, count=1, flags=re.IGNORECASE)
        return cleaned, cleaned != cypher

    @staticmethod
    def boost_vector_candidate_count(cypher: str, min_k: int = 50) -> tuple:
        """
        Increase the candidate count in queryNodes(..., k, ...) for vector queries.

        The chunkEmbeddingIndex mixes fund-profile chunks and company-filing chunks.
        A low k (e.g. 10–15) leaves too few company-filing candidates after the
        post-YIELD MATCH filter, causing relevant chunks to fall outside the window.
        Bumps k to at least `min_k` when the current value is lower.

        Returns (cypher: str, was_modified: bool).
        """
        if 'db.index.vector.queryNodes' not in cypher:
            return cypher, False

        def _bump(m):
            current_k = int(m.group(1))
            if current_k < min_k:
                return m.group(0).replace(m.group(1), str(min_k), 1)
            return m.group(0)

        pattern = re.compile(
            r"db\.index\.vector\.queryNodes\s*\(\s*'[^']+'\s*,\s*(\d+)\s*,",
            re.IGNORECASE,
        )
        cleaned = pattern.sub(_bump, cypher)
        return cleaned, cleaned != cypher

    @staticmethod
    def fix_portfolio_count_ordering(cypher: str) -> tuple:
        """
        Fix holdings queries that sort by p.count (total portfolio size) instead of
        r.weight (individual holding weight).

        p.count is the same value for every holding in a portfolio, so ORDER BY p.count
        produces arbitrary ordering. "Top holdings" always means ORDER BY r.weight DESC.

        Transforms:
          -[:HAS_HOLDING]->(h:Holding) ... ORDER BY p.count DESC
          → -[r:HAS_HOLDING]->(h:Holding) ... ORDER BY r.weight DESC

        Returns (fixed_cypher: str, was_modified: bool).
        """
        if 'HAS_HOLDING' not in cypher:
            return cypher, False
        if not re.search(r'\bORDER\s+BY\s+\w+\.count\b', cypher, re.IGNORECASE):
            return cypher, False
        # Only fix when a specific fund/company is targeted — without a filter p.count
        # differs per portfolio and ordering by it is legitimate.
        if not re.search(r':Fund\s*\{', cypher, re.IGNORECASE):
            return cypher, False

        was_modified = False

        # Inject named variable on HAS_HOLDING if anonymous
        if re.search(r'-\[:HAS_HOLDING\]->', cypher, re.IGNORECASE):
            cypher = re.sub(r'-\[:HAS_HOLDING\]->', '-[r:HAS_HOLDING]->', cypher, count=1, flags=re.IGNORECASE)
            was_modified = True

        # Determine variable name on HAS_HOLDING
        m = re.search(r'\[(\w+):HAS_HOLDING\]', cypher, re.IGNORECASE)
        holding_var = m.group(1) if m else 'r'

        # Replace ORDER BY <any_var>.count with ORDER BY <holding_var>.weight DESC
        cleaned, n = re.subn(
            r'\bORDER\s+BY\s+\w+\.count\b(\s+(?:ASC|DESC))?',
            f'ORDER BY {holding_var}.weight DESC',
            cypher,
            flags=re.IGNORECASE,
        )
        if n > 0:
            cypher = cleaned
            was_modified = True

        return cypher, was_modified

    @staticmethod
    def inject_year_in_section_query(cypher: str) -> tuple:
        """
        For structural 10-K section queries (REPORTS_IN → HAS_SECTION → HAS_CHUNK),
        ensure r.year is included in RETURN and ORDER BY.

        Without r.year the results from multiple filing years are mixed with no
        way to distinguish them and the most recent year's chunks appear last.

        Transforms:
          (c:Company)-[:REPORTS_IN]->(f:Filing10K)-...->(chunk:Chunk)
          RETURN s.title, chunk.text ORDER BY chunk.id ASC
          →
          (c:Company)-[r:REPORTS_IN]->(f:Filing10K)-...->(chunk:Chunk)
          RETURN s.title, chunk.text, r.year AS year ORDER BY r.year DESC, chunk.id ASC

        Returns (fixed_cypher: str, was_modified: bool).
        """
        if 'db.index.vector.queryNodes' in cypher:
            return cypher, False
        if 'REPORTS_IN' not in cypher:
            return cypher, False
        if 'HAS_CHUNK' not in cypher:
            return cypher, False
        if not re.search(r'chunk\.text', cypher, re.IGNORECASE):
            return cypher, False

        was_modified = False

        # Find or inject REPORTS_IN variable
        m = re.search(r'\[(\w+):REPORTS_IN\]', cypher, re.IGNORECASE)
        if m:
            reports_var = m.group(1)
        else:
            cypher = re.sub(r'-\[:REPORTS_IN\]->', '-[r:REPORTS_IN]->', cypher, count=1, flags=re.IGNORECASE)
            reports_var = 'r'
            was_modified = True

        # Skip if r.year already in query
        if re.search(rf'\b{re.escape(reports_var)}\.year\b', cypher, re.IGNORECASE):
            return cypher, was_modified

        # Add r.year AS year to RETURN (before ORDER BY / LIMIT / end)
        cypher_new = re.sub(
            r'(RETURN\s+.+?)(\s+ORDER\s+BY|\s+LIMIT|\s*$)',
            lambda m2: m2.group(1) + f', {reports_var}.year AS year' + m2.group(2),
            cypher,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cypher_new != cypher:
            cypher = cypher_new
            was_modified = True

        # Prefix r.year DESC to ORDER BY if not already leading
        order_m = re.search(r'\bORDER\s+BY\s+', cypher, re.IGNORECASE)
        if order_m:
            after_order = cypher[order_m.end():]
            if not re.match(rf'{re.escape(reports_var)}\.year', after_order, re.IGNORECASE):
                cypher = re.sub(
                    r'\bORDER\s+BY\s+',
                    f'ORDER BY {reports_var}.year DESC, ',
                    cypher,
                    count=1,
                    flags=re.IGNORECASE,
                )
                was_modified = True

        return cypher, was_modified

    @staticmethod
    def inject_fund_provider_trust(cypher: str) -> tuple[str, bool]:
        """Append Provider and Trust to Fund queries that don't already include them.

        When a query returns properties of a specific Fund but omits who issues it,
        this injects an OPTIONAL MATCH for the Provider→Trust→Fund chain and appends
        prov.name AS provider, t.name AS trust to the RETURN clause.

        Guards — skips when any of these are true:
        - Vector / fulltext CALL queries
        - :Provider or :Trust already in query
        - Aggregation functions (COUNT, SUM, AVG, MIN, MAX) — would break GROUP BY
        - No specific Fund filter ({ticker:} or {name:}) — global scans excluded
        - More than one WITH clause — complex pipelines excluded
        - DISTINCT across all columns — cardinality sensitive
        """
        if 'db.index.vector.queryNodes' in cypher or 'db.index.fulltext.queryNodes' in cypher:
            return cypher, False
        if ':Fund' not in cypher:
            return cypher, False
        # Provider or Trust already traversed
        if re.search(r'\b:Provider\b|\b:Trust\b', cypher):
            return cypher, False
        # provider/trust already returned under any alias
        if re.search(r'\bprov(?:ider)?\b\s*\.|t\.name\b|\btrustName\b|\bproviderName\b', cypher, re.IGNORECASE):
            return cypher, False
        # skip aggregations
        if re.search(r'\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(', cypher, re.IGNORECASE):
            return cypher, False
        # require a specific fund filter — don't fire on global scans
        if not re.search(r':Fund\s*\{\s*(?:ticker|name)\s*:', cypher, re.IGNORECASE):
            return cypher, False
        # skip complex multi-WITH pipelines
        if len(re.findall(r'\bWITH\b', cypher, re.IGNORECASE)) > 1:
            return cypher, False

        fund_m = re.search(r'\(\s*(\w+)\s*:Fund\b', cypher, re.IGNORECASE)
        if not fund_m:
            return cypher, False
        fund_var = fund_m.group(1)

        return_m = re.search(r'\bRETURN\b', cypher, re.IGNORECASE)
        if not return_m:
            return cypher, False

        before_return = cypher[:return_m.start()].rstrip()
        return_and_after = cypher[return_m.start():]

        # Split RETURN columns from trailing ORDER BY / LIMIT / SKIP / UNION
        trailer_m = re.search(r'\b(?:ORDER\s+BY|LIMIT|SKIP|UNION)\b', return_and_after, re.IGNORECASE)
        if trailer_m:
            return_block = return_and_after[:trailer_m.start()].rstrip()
            trailer = "\n" + return_and_after[trailer_m.start():]
        else:
            return_block = return_and_after.rstrip()
            trailer = ""

        opt_match = f"\nOPTIONAL MATCH (prov:Provider)-[:MANAGES]->(t:Trust)-[:ISSUES]->({fund_var})"
        new_cypher = (
            before_return
            + opt_match
            + "\n"
            + return_block
            + ", prov.name AS provider, t.name AS trust"
            + trailer
        )
        return new_cypher, True

    def check_syntax_only(self, query: str) -> ValidationResult:
        """Only check Cypher syntax, no schema validation."""
        result = ValidationResult(is_valid=True, original_query=query)
        if not query or not query.strip():
            result.is_valid = False
            result.syntax_errors.append("Empty query")
            return result

        if not self._check_syntax(query.strip(), result):
            result.is_valid = False
        self._classify_query(query.strip(), result)
        return result

    def _check_syntax(self, query: str, result: ValidationResult) -> bool:
        """
        Run Neo4j EXPLAIN to validate cypher syntax and schema presence.
        """
        if not self.driver:
            return True
            
        try:
            with self.driver.session() as session:
                explain_query = f"EXPLAIN {query}"
                session.run(explain_query)
            return True
        except Exception as e:
            error_name = type(e).__name__
            result.syntax_errors.append(f"{error_name}: {e}")
            result.triggered_rule = "NEO4J_SYNTAX"
            return False

    def _classify_query(self, query: str, result: ValidationResult):
        """
        Classify query as read or write using keyword detection.

        Write operations are detected by looking for keywords that modify data:
        - CREATE / CREATE() - Creates new nodes or relationships
        - MERGE / MERGE() - Creates or updates nodes/relationships
        - DELETE / DETACH DELETE - Deletes nodes or relationships
        - SET - Updates node/relationship properties
        - REMOVE - Removes node/relationship properties or labels
        - DROP - Drops indexes or constraints
        - ALTER - Modifies database structure

        Only MATCH, RETURN, WITH, WHERE, ORDER BY, LIMIT, SKIP are allowed (read-only).
        """
        upper = query.upper()
        result.is_write_query = any(kw in upper for kw in self.BLOCKED_WRITE_KEYWORDS)
        result.is_read_query = not result.is_write_query

    def _check_undefined_relationship_variables(self, query: str) -> Dict[str, List[str]]:
        """
        Check if relationship variables are used but not defined.

        Returns a dict of {variable_name: [property1, property2, ...]} for undefined vars.
        Example: {'r': ['weight', 'year']} means r.weight and r.year are used but r is not defined.
        """
        # Extract all relationship variables defined in MATCH patterns: [r:REL_TYPE]
        defined_rel_vars = set()
        rel_def_pattern = re.compile(r'\[\s*([a-zA-Z_]\w*)\s*:\s*[A-Z_]')
        for match in rel_def_pattern.finditer(query):
            defined_rel_vars.add(match.group(1))

        # Extract all node variables defined in MATCH patterns: (varName:Label)
        # If a variable is bound to a node, it should never be flagged as an undefined rel var
        defined_node_vars = set(re.findall(r'\(\s*([a-zA-Z_]\w*)\s*:[A-Za-z_]', query))

        # Extract all relationship variable usages: r.property, r.year, etc.
        # Look in RETURN, WHERE, ORDER BY, WITH clauses
        used_rel_vars = {}
        rel_usage_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')

        for match in rel_usage_pattern.finditer(query):
            var_name = match.group(1)
            property_name = match.group(2)

            # Only flag properties that are exclusively on relationships (not nodes)
            if property_name in {'weight', 'year', 'date', 'ceoCompensation', 'ceoActuallyPaid',
                                'marketValue', 'currency', 'fairValueLevel',
                                'isRestricted', 'payoffProfile'}:
                # Skip if already defined as a relationship variable
                if var_name in defined_rel_vars:
                    continue
                # Skip if the variable is bound to a node — it's a node property access, not a rel var
                if var_name in defined_node_vars:
                    continue
                if var_name not in used_rel_vars:
                    used_rel_vars[var_name] = []
                used_rel_vars[var_name].append(property_name)

        return used_rel_vars

    def _validate_schema_lightweight(self, query: str, result: ValidationResult):
        """
        Lightweight schema validation using regex extraction.
        
        Checks:
        - Node labels used in the query exist in the schema
        - Relationship types used in the query exist in the schema
        """
        # Extract node labels: (var:Label) or (:Label)
        label_pattern = re.compile(r'\(\s*\w*\s*:\s*(`[^`]+`|[A-Za-z_]\w*)')
        for match in label_pattern.finditer(query):
            label = match.group(1).strip('`')
            if label not in self.VALID_LABELS:
                result.schema_errors.append(f"Unknown node label: '{label}'")

        # Extract relationship types: [:REL_TYPE] or [r:REL_TYPE]
        rel_pattern = re.compile(r'\[\s*\w*\s*:\s*(!?\s*`[^`]+`|!?\s*[A-Z_][A-Z_0-9]*)')
        for match in rel_pattern.finditer(query):
            rel_type = match.group(1).strip().strip('`').strip('!')
            if rel_type and rel_type not in self.VALID_REL_TYPES:
                result.schema_errors.append(f"Unknown relationship type: '{rel_type}'")


# ─── Result-level validator ───────────────────────────────────────────────────

class ResultValidator:
    """
    Post-execution semantic checks on query results.

    These are data-quality rules that cannot be caught at query-parse time
    because they depend on what Neo4j actually returns.  Each failing check
    produces a ValidationResult with a targeted error message that the retry
    loop can feed back to the LLM.
    """

    @staticmethod
    def validate(cypher: str, records: list) -> Optional["ValidationResult"]:
        """
        Run all result-level checks.  Returns the first failing ValidationResult,
        or None if every check passes.

        Args:
            cypher:  The Cypher query that produced the records.
            records: List of dicts returned by Neo4j (may be empty).
        """
        for check in (
            ResultValidator._check_zero_expense_ratio,
        ):
            result = check(cypher, records)
            if result is not None:
                return result
        return None

    # ── Individual checks ────────────────────────────────────────────────────

    @staticmethod
    def _check_zero_expense_ratio(cypher: str, records: list) -> Optional["ValidationResult"]:
        """
        Trigger when the query asks for expenseRatio but every returned value
        is 0.0 (or None).  This indicates missing data-quality guard.
        """
        if not records:
            return None

        # Only apply when the query returns expenseRatio from FinancialHighlight
        cypher_upper = cypher.upper()
        if 'EXPENSERATIO' not in cypher_upper:
            return None
        # Guard already present — no action needed
        if re.search(r'fh\.expenseRatio\s*>\s*0', cypher, re.IGNORECASE):
            return None

        # Collect every expenseRatio value from the result set
        ratio_values = []
        for row in records:
            for key, val in row.items():
                if 'expenseratio' in key.lower():
                    ratio_values.append(val)

        if not ratio_values:
            return None

        # All values are 0.0 / None — data quality issue
        all_zero = all(v is None or v == 0.0 or v == 0 for v in ratio_values)
        if not all_zero:
            return None

        result = ValidationResult(is_valid=False, original_query=cypher)
        result.triggered_rule = "ZERO_EXPENSE_RATIO"
        result.schema_errors.append(
            "ZERO EXPENSE RATIO ERROR: Every expenseRatio value returned is 0.0. "
            "This means your query is matching FinancialHighlight rows where the expense ratio "
            "was not recorded (stored as 0.0 in the database). "
            "Add WHERE fh.expenseRatio > 0 (or AND fh.expenseRatio > 0 if a WHERE clause already exists) "
            "to filter out these placeholder rows and return only real expense ratio data. "
            "BAD:  MATCH (f:Fund {ticker: 'VFINX'})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) "
            "RETURN fh.expenseRatio ORDER BY r.year DESC LIMIT 1 "
            "GOOD: MATCH (f:Fund {ticker: 'VFINX'})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) "
            "WHERE fh.expenseRatio > 0 RETURN fh.expenseRatio ORDER BY r.year DESC LIMIT 1"
        )
        return result


# ─── Convenience Functions ───────────────────────────────────────────────────

_default_validator: Optional[CypherValidator] = None


def get_validator(neo4j_driver=None, block_writes: bool = True) -> CypherValidator:
    """Get or create the singleton CypherValidator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = CypherValidator(neo4j_driver=neo4j_driver, block_writes=block_writes)
    return _default_validator


def validate_cypher(query: str, neo4j_driver=None) -> ValidationResult:
    """
    Convenience function to validate a Cypher query.

    Example:
        >>> from simple_rag.rag.post_processing.cypher_validator import validate_cypher
        >>> result = validate_cypher("MATCH (f:Fund {ticker: 'VTI'}) RETURN f.name")
        >>> result.is_valid
        True
    """
    return get_validator(neo4j_driver).validate(query)


def is_read_only_query(query: str) -> bool:
    """
    Check if a query is read-only (no write operations).

    Returns:
        True if query contains no write operations (CREATE, DELETE, MERGE, SET, etc.)
        False if query would modify data

    Example:
        >>> is_read_only_query("MATCH (f:Fund) RETURN f.name")
        True
        >>> is_read_only_query("CREATE (f:Fund) RETURN f")
        False
    """
    validator = CypherValidator(block_writes=True)
    temp_result = ValidationResult(is_valid=True, original_query=query)
    validator._classify_query(query, temp_result)
    # Return True if it's a read query (no write detected)
    return not temp_result.is_write_query