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
        "InsiderTransaction": {"position", "transactionType", "shares", "price",
                               "value", "remainingShares"},
        "Chunk": {"title", "text", "embedding", "chunkType", "chunkIndex",
                  "subsection", "sectionType", "sectionName", "ticker", "filingDate",
                  "wordCount"},
        "SectionChunk": {"title", "text", "embedding", "chunkType", "chunkIndex",
                         "subsection", "sectionName", "ticker", "filingDate"},
        "Filing10K": set(),
        "Section": {"title", "text", "embedding", "sectionType", "secItem"},
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
                        ("Profile", "PerformanceCommentary"), ("Profile", "RiskFactor"),
                        ("Profile", "Strategy")},
        "HAS_FINACIALS": {("Filing10K", "Financials")},  # typo preserved from schema
        "HAS_FINANCIALS": {("Filing10K", "Financials")},
        "HAS_METRIC": {("Financials", "FinancialMetric")},
        "HAS_SEGMENT": {("FinancialMetric", "Segment")},
        "EXTRACTED_FROM": {("Fund", "Document"), ("Profile", "Document"),
                           ("Portfolio", "Document"), ("Filing10K", "Document"),
                           ("InsiderTransaction", "Document")},
        "HAS_CHUNK": {("Section", "Chunk"), ("Section", "SectionChunk"),
                      ("RiskFactor", "Chunk"), ("RiskFactor", "SectionChunk"),
                      ("BusinessInformation", "Chunk"), ("BusinessInformation", "SectionChunk"),
                      ("LegalProceeding", "Chunk"), ("LegalProceeding", "SectionChunk"),
                      ("ManagementDiscussion", "Chunk"), ("ManagementDiscussion", "SectionChunk"),
                      ("Properties", "Chunk"), ("Properties", "SectionChunk"),
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
            r'\bWHERE\b[^;]*?\b(COUNT|SUM|AVG|MIN|MAX)\s*\(',
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

        # Extract all relationship variable usages: r.property, r.year, etc.
        # Look in RETURN, WHERE, ORDER BY, WITH clauses
        used_rel_vars = {}
        rel_usage_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')

        for match in rel_usage_pattern.finditer(query):
            var_name = match.group(1)
            property_name = match.group(2)

            # Skip node variables (f, g, h, p, c, etc.) - these are typically single letters
            # We're looking for relationship variables that are used but not defined
            # Check if this variable is used in a context that suggests it's a relationship variable
            # (i.e., used for properties that are typically on relationships like weight, year)
            if property_name in {'weight', 'year', 'date', 'ceoCompensation', 'ceoActuallyPaid',
                                'shares', 'marketValue', 'currency', 'fairValueLevel',
                                'isRestricted', 'payoffProfile'}:
                if var_name not in defined_rel_vars:
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