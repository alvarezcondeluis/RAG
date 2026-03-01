"""
Cypher Query Validator Module.

Uses cypher-guard for syntax checking (check_syntax, is_read, is_write)
and implements custom lightweight schema validation.

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
        "Fund": {"ticker", "name", "securityExchange", "costsPer10k", "advisoryFees",
                 "numberHoldings", "expenseRatio", "netAssets", "turnoverRate", "updatedAt"},
        "ShareClass": {"name", "description"},
        "Profile": {"id", "summaryProspectus"},
        "Objective": {"id", "text", "embedding"},
        "RiskChunk": {"id", "title", "text", "embedding"},
        "StrategyChunk": {"id", "title", "text", "embedding"},
        "PerformanceCommentary": {"id", "text", "embedding"},
        "Image": {"id", "title", "category", "svg"},
        "Sector": {"name"},
        "GeographicAllocation": {"name"},
        "Person": {"name"},
        "Portfolio": {"id", "ticker", "count", "seriesId", "date"},
        "Holding": {"id", "name", "ticker", "cusip", "isin", "lei", "country",
                    "sector", "assetCategory", "assetDesc", "issuerCategory", "issuerDesc"},
        "FinancialHighlight": {"id", "turnover", "expenseRatio", "totalReturn",
                               "netAssets", "netAssetsValueBeginning", "netAssetsValueEnd",
                               "netIncomeRatio"},
        "TrailingPerformance": {"return1y", "return5y", "return10y", "returnInception"},
        "Company": {"ticker", "name", "cik"},
        "CompensationPackage": {"totalCompensation", "shareholderReturn", "date"},
        "InsiderTransaction": {"position", "transactionType", "shares", "price",
                               "value", "remainingShares"},
        "Filing10K": {"id"},
        "Section": {"id", "text", "embedding"},
        "RiskFactor": {"id", "text", "embedding"},
        "BusinessInformation": {"id", "text", "embedding"},
        "LegalProceeding": {"id", "text", "embedding"},
        "ManagemetDiscussion": {"id", "text", "embedding"},
        "Properties": {"id", "text", "embedding"},
        "Financials": {"incomeStatement", "balanceSheet", "cashFlow", "fiscalYear"},
        "FinancialMetric": {"label", "value"},
        "Segment": {"label", "value", "percentage"},
        "Document": {"id", "accession_number", "url", "form", "filing_date",
                     "reporting_date", "accessionNumber", "filingDate", "reportingDate", "type"},
    }

    # ─── Schema: valid relationship types and their (start, end) pairs ───
    RELATIONSHIPS: Dict[str, Set[Tuple[str, str]]] = {
        "MANAGES": {("Provider", "Trust")},
        "ISSUES": {("Trust", "Fund")},
        "HAS_SHARE_CLASS": {("Fund", "ShareClass")},
        "DEFINED_BY": {("Fund", "Profile")},
        "HAS_OBJECTIVE": {("Profile", "Objective")},
        "HAS_RISK_NODE": {("Profile", "RiskChunk")},
        "HAS_STRATEGY": {("Profile", "StrategyChunk")},
        "HAS_PERFORMANCE_COMMENTARY": {("Profile", "PerformanceCommentary")},
        "HAS_CHART": {("Fund", "Image")},
        "HAS_SECTOR_ALLOCATION": {("Fund", "Sector")},
        "HAS_GEOGRAPHIC_ALLOCATION": {("Fund", "Region")},
        "MANAGED_BY": {("Fund", "Person")},
        "HAS_PORTFOLIO": {("Fund", "Portfolio")},
        "HAS_HOLDING": {("Portfolio", "Holding")},
        "HAS_FINANCIAL_HIGHLIGHT": {("Fund", "FinancialHighlight")},
        "HAS_TRAILING_PERFORMANCE": {("Fund", "TrailingPerformance")},
        "REPRESENTS": {("Holding", "Company")},
        "EMPLOYED_AS_CEO": {("Company", "Person")},
        "AWARDED_BY": {("Company", "CompensationPackage")},
        "RECEIVED_COMPENSATION": {("Person", "CompensationPackage")},
        "HAS_INSIDER_TRANSACTION": {("Company", "InsiderTransaction")},
        "MADE_BY": {("InsiderTransaction", "Person")},
        "HAS_FILING": {("Company", "Filing10K")},
        "HAS_RISK_FACTORS": {("Filing10K", "RiskFactor")},
        "HAS_BUSINESS_INFORMATION": {("Filing10K", "BusinessInformation")},
        "HAS_LEGAL_PROCEEDINGS": {("Filing10K", "LegalProceeding")},
        "HAS_MANAGEMENT_DISCUSSION": {("Filing10K", "ManagemetDiscussion")},
        "HAS_PROPERTIES_CHUNK": {("Filing10K", "Properties")},
        "HAS_FINACIALS": {("Filing10K", "Financials")},
        "HAS_FINANCIALS": {("Filing10K", "Financials")},
        "HAS_METRIC": {("Financials", "FinancialMetric")},
        "HAS_SEGMENT": {("FinancialMetric", "Segment")},
        "EXTRACTED_FROM": {("Fund", "Document"), ("Profile", "Document"),
                           ("Portfolio", "Document"), ("Filing10K", "Document")},
        "DISCLOSED_IN": {("InsiderTransaction", "Document"),
                         ("CompensationPackage", "Document")},
    }

    # Flat sets for quick lookup
    VALID_REL_TYPES: Set[str] = set(RELATIONSHIPS.keys())
    VALID_LABELS: Set[str] = set(NODE_PROPERTIES.keys())

    # Relationship properties
    REL_PROPERTIES: Dict[str, Set[str]] = {
        "DEFINED_BY": {"date"},
        "HAS_CHART": {"date"},
        "HAS_SECTOR_ALLOCATION": {"weight", "reportDate", "date"},
        "HAS_GEOGRAPHIC_ALLOCATION": {"weight", "reportDate", "date"},
        "MANAGED_BY": {"date"},
        "HAS_HOLDING": {"shares", "marketValue", "weight", "currency",
                        "fairValueLevel", "isRestricted", "payoffProfile"},
        "HAS_FINANCIAL_HIGHLIGHT": {"year"},
        "HAS_TRAILING_PERFORMANCE": {"date"},
        "EMPLOYED_AS_CEO": {"ceoCompensation", "ceoActuallyPaid", "date"},
        "HAS_FILING": {"date"},
        "HAS_PORTFOLIO": {"date"},
    }

    def __init__(self, neo4j_driver=None, block_writes: bool = True, use_syntax_check: bool = True):
        """
        Initialize the CypherValidator.
        
        Args:
            neo4j_driver: Neo4j driver instance to run EXPLAIN queries for validation.
            block_writes: If True, mark write queries as invalid.
            use_syntax_check: If True, use Neo4j EXPLAIN for syntax checking.
                             Set to False to skip syntax check (faster, less noise).
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

        # Step 0: Fast pre-check — catch inline math operators inside {} before Neo4j
        inline_math = re.search(r'\{[^}]*(?:[<>]=?|!=)\s*[\d\w\'"]', query)
        if inline_math:
            result.is_valid = False
            result.syntax_errors.append(
                "INLINE MATH ERROR: You used a comparison operator (>, <, >=, <=, !=) "
                "inside curly braces in a MATCH pattern. This is invalid Cypher syntax. "
                "Move the condition to a WHERE clause after MATCH. "
                "BAD:  MATCH (n:Node {prop: > 10}) "
                "GOOD: MATCH (n:Node)-[r:REL]->(m) WHERE r.prop > 10"
            )
            return result

        # Step 1: Write detection (keyword-based, no cypher-guard needed)
        self._classify_query(query, result)
        if self.block_writes and result.is_write_query:
            result.is_valid = False
            result.syntax_errors.append(
                "Write operations (CREATE, MERGE, DELETE, SET) are not allowed"
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
            return False

    def _classify_query(self, query: str, result: ValidationResult):
        """Classify query as read or write using keyword detection."""
        upper = query.upper()
        write_keywords = ["CREATE ", "CREATE(", "MERGE ", "MERGE(", 
                          "DELETE ", "SET ", "REMOVE ", "DROP "]
        result.is_write_query = any(kw in upper for kw in write_keywords)
        result.is_read_query = not result.is_write_query

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