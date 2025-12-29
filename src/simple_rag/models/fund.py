from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd


@dataclass
class AverageReturnSnapshot:

    return_1y: str
    return_5y: str
    return_10y: str
    return_inception: str

@dataclass
class FinancialHighlights:
    turnover: float
    expense_ratio: float
    total_return: float
    net_assets: float
    net_assets_value_begining: float
    net_assets_value_end: float
    net_income_ratio: float

@dataclass
class FundData:
    """Defines the structure of the output. No logic here."""
    name: str
    registrant: str
    context_id: str
    share_class: Optional[str] = None
    ticker: str = "N/A"
    costs_per_10k: str = "N/A"
    expense_ratio: str = "N/A"
    report_date: str = "N/A"
    net_assets: str = "N/A"
    security_exchange: Optional[str] = None
    turnover_rate: str = "N/A"
    advisory_fees: str = "N/A"
    n_holdings: str = "N/A"
    annual_returns: Optional[Dict[str,float]] = None
    performance: Optional[Dict[str, AverageReturnSnapshot]] = field(default_factory=dict)
    financial_highlights: Optional[Dict[str, FinancialHighlights]] = None
    managers: Optional[List[str]] = field(default_factory=list)
    performance_commentary: str = "N/A"
    summary_prospectus: str = "N/A"
    strategies: str = "N/A"
    risks: str = "N/A"
    objective: str = "N/A"
    avg_annual_returns: Optional[pd.DataFrame] = None
    # Dataframes can be stored directly, or converted to string/dicts
    geographic_allocation: Optional[pd.DataFrame] = None
    top_holdings: Optional[pd.DataFrame] = None
    portfolio_composition: Optional[pd.DataFrame] = None
    sector_allocation: Optional[pd.DataFrame] = None
    performance_table: Optional[pd.DataFrame] = None
    maturity_allocation: Optional[pd.DataFrame] = None
    credit_rating: Optional[pd.DataFrame] = None
    issuer_allocation: Optional[pd.DataFrame] = None
