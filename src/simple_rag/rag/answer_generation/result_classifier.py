"""
Lightweight deterministic classifier for Neo4j result types.

Inspects the keys and values in result dictionaries to determine the data type
(financial metrics, holdings, text, charts, etc.) for proper rendering.
"""

from enum import Enum
from typing import Any, Dict, List


class ResultType(str, Enum):
    FINANCIAL_METRICS = "financial_metrics"
    HOLDINGS_TABLE = "holdings_table"
    TEXT_CHUNKS = "text_chunks"
    CHART_SVG = "chart_svg"
    COMPANY_FINANCIALS = "company_financials"
    ALLOCATION_DATA = "allocation_data"
    PEOPLE_COMPENSATION = "people_compensation"
    EMPTY = "empty"
    GENERIC = "generic"


# Key sets for heuristic detection
_SVG_KEYS = {"svg", "chart", "image"}
_HOLDINGS_KEYS = {"shares", "marketValue", "weight", "isin", "lei"}
_TEXT_KEYS = {"text", "embedding", "fullText"}
_FINANCIAL_KEYS = {"expenseRatio", "totalReturn", "netAssets", "turnoverRate", "costsPer10k"}
_HIGHLIGHT_KEYS = {"turnover", "netIncomeRatio", "netAssetsValueBeginning", "netAssetsValueEnd"}
_PEOPLE_KEYS = {"ceoCompensation", "totalCompensation", "ceoActuallyPaid", "transactionType"}
_ALLOCATION_KEYS = {"weight", "name"}
_COMPANY_FIN_KEYS = {"incomeStatement", "balanceSheet", "cashFlow", "fiscalYear"}

# Category -> ResultType fallback mapping
_CATEGORY_MAP = {
    "fund_basic": ResultType.FINANCIAL_METRICS,
    "fund_portfolio": ResultType.HOLDINGS_TABLE,
    "fund_profile": ResultType.TEXT_CHUNKS,
    "company_filing": ResultType.TEXT_CHUNKS,
    "company_people": ResultType.PEOPLE_COMPENSATION,
}


class ResultClassifier:
    """Classifies Neo4j query results by data type for rendering."""

    def classify(
        self,
        results: List[Dict[str, Any]],
        query_category: str = "",
    ) -> ResultType:
        """Classify the result type based on keys present in the data.

        Args:
            results: List of Neo4j result dictionaries.
            query_category: The SetFit query category (e.g. "fund_basic").

        Returns:
            The detected ResultType.
        """
        if not results:
            return ResultType.EMPTY

        # Collect all keys across all result dicts
        all_keys = set()
        for row in results:
            all_keys.update(row.keys())

        # Priority-based detection
        if all_keys & _SVG_KEYS:
            return ResultType.CHART_SVG

        if all_keys & _HOLDINGS_KEYS - {"weight"}:
            # Need more than just "weight" to distinguish from allocations
            return ResultType.HOLDINGS_TABLE

        if all_keys & _TEXT_KEYS:
            return ResultType.TEXT_CHUNKS

        if all_keys & _PEOPLE_KEYS:
            return ResultType.PEOPLE_COMPENSATION

        if all_keys & _COMPANY_FIN_KEYS:
            return ResultType.COMPANY_FINANCIALS

        if all_keys & _FINANCIAL_KEYS or all_keys & _HIGHLIGHT_KEYS:
            return ResultType.FINANCIAL_METRICS

        # "weight" + "name" without holdings keys -> allocation
        if _ALLOCATION_KEYS.issubset(all_keys) and not (all_keys & _HOLDINGS_KEYS - {"weight"}):
            return ResultType.ALLOCATION_DATA

        # Fallback to category mapping
        if query_category in _CATEGORY_MAP:
            return _CATEGORY_MAP[query_category]

        return ResultType.GENERIC

    def extract_svg_data(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract SVG chart data from results.

        Returns:
            List of dicts with 'svg', 'title', and 'category' keys.
        """
        charts = []
        for row in results:
            if "svg" in row and row["svg"]:
                charts.append({
                    "svg": row["svg"],
                    "title": row.get("title", "Chart"),
                    "category": row.get("category", ""),
                })
        return charts

    def extract_tabular_data(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return results suitable for table rendering (strip embeddings, SVGs)."""
        skip_keys = {"embedding", "svg"}
        cleaned = []
        for row in results:
            cleaned.append({k: v for k, v in row.items() if k not in skip_keys})
        return cleaned
