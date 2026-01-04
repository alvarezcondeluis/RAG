from typing import Optional, Dict, List
from decimal import Decimal
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class AverageReturnSnapshot(BaseModel):
    """Snapshot of fund returns over different time periods."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    return_1y: str = Field(description="1-year return percentage from the relationship date")
    return_5y: str = Field(description="5-year return percentage from the relationship date")
    return_10y: str = Field(description="10-year return percentage from the relationship date")
    return_inception: str = Field(description="Return since fund inception (percentage)")

class PortfolioHolding(BaseModel):
    """Simplified holding for fund analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(description="Security name")
    cusip: str = Field(description="CUSIP identifier")
    isin: str = Field(description="ISIN identifier")
    shares: Decimal = Field(description="Number of shares held")
    market_value: Decimal = Field(description="Market value of the holding")
    weight_pct: Decimal = Field(description="Percentage weight in portfolio")
    asset_category: str = Field(description="Asset category classification")
    country: str = Field(description="Country of issuance")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    lei: Optional[str] = Field(None, description="Legal Entity Identifier")
    asset_category_desc: Optional[str] = Field(None, description="Asset category description")
    issuer_category: Optional[str] = Field(None, description="Issuer category code")
    issuer_category_desc: Optional[str] = Field(None, description="Issuer category description")
    currency: Optional[str] = Field(None, description="Currency denomination")
    payoff_profile: Optional[str] = Field(None, description="Derivative payoff profile")
    is_restricted: bool = Field(False, description="Whether the security is restricted")
    fair_value_level: Optional[str] = Field(None, description="Fair value hierarchy level (1, 2, or 3)")
    sector: Optional[str] = Field(None, description="Industry sector classification")

class FinancialHighlights(BaseModel):
    """Annual financial highlights and metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    turnover: float = Field(description="Portfolio turnover rate (percentage)")
    expense_ratio: float = Field(description="Total expense ratio (percentage)")
    total_return: float = Field(description="Total return for the period (percentage)")
    net_assets: float = Field(description="Total net assets under management (in millions) dollars")
    net_assets_value_begining: float = Field(description="Price of one share at period start (in dollars)")
    net_assets_value_end: float = Field(description="Price of one share at the end of the period (in dollars)")
    net_income_ratio: float = Field(description=(
        "The ratio of net investment income to average net assets, expressed as a percentage. "
        "This serves as a proxy for the fund's annual dividend yield. "
        "Higher values indicate an income-focused strategy; lower values indicate a growth focus."
    ))

class Derivatives(BaseModel):
    """Derivative holdings information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    date: str = Field(description="Report date for derivatives data")
    derivatives_df: Optional[pd.DataFrame] = Field(None, description="DataFrame containing derivative positions")


class NonDerivatives(BaseModel):
    """Non-derivative holdings information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    date: str = Field(description="Report date for holdings data")
    holdings_df: Optional[pd.DataFrame] = Field(None, description="DataFrame containing non-derivative holdings")

class ShareClassType(str, Enum):
    ADMIRAL = "Admiral Shares"
    INVESTOR = "Investor Shares"
    INSTITUTIONAL = "Institutional Shares"
    INSTITUTIONAL_PLUS = "Institutional Plus Shares"
    INSTITUTIONAL_SELECT = "Institutional Select Shares"
    ETF = "ETF Shares"
    OTHER = "Other"

class FundData(BaseModel):
    """Complete fund data structure with all fund information and metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Core identifiers
    name: str = Field(description="Official fund name")
    registrant: str = Field(description="Registered investment company name")
    context_id: str = Field(description="SEC EDGAR context identifier")
    share_class: ShareClassType = Field(None, description="Share class designation (e.g., Admiral, Investor)")
    ticker: str = Field("N/A", description="Stock ticker symbol")
    security_exchange: Optional[str] = Field(None, description="Primary listing exchange")
    
    # Financial metrics
    costs_per_10k: str = Field("N/A", description="Cost per $10,000 invested")
    expense_ratio: str = Field("N/A", description="Annual expense ratio as percentage (Annual fee charged by the fund)")
    net_assets: str = Field("N/A", description="Total net assets under management in millions")
    turnover_rate: str = Field("N/A", description="percentage of a fund's holdings that have been replaced (bought and sold) over a one-year period")
    advisory_fees: str = Field("N/A", description="Investment advisory fees")
    n_holdings: str = Field("N/A", description="Total number of holdings in portfolio")
    
    # Dates
    report_date: str = Field("N/A", description="Date of the report or filing")
    
    # Performance data
    annual_returns: Optional[Dict[str, float]] = Field(None, description="Annual returns by year")
    performance: Optional[Dict[str, AverageReturnSnapshot]] = Field(
        default_factory=dict,
        description="Performance snapshots for different time periods"
    )
    avg_annual_returns: Optional[pd.DataFrame] = Field(None, description="Average annual returns table")
    performance_table: Optional[pd.DataFrame] = Field(None, description="Detailed performance metrics table")
    
    # Financial highlights
    financial_highlights: Optional[Dict[str, FinancialHighlights]] = Field(
        default_factory=dict,
        description="Annual financial highlights by year"
    )
    
    # Management
    managers: Optional[List[str]] = Field(
        default_factory=list,
        description="List of portfolio managers"
    )
    
    # Fund strategy and risk
    performance_commentary: str = Field("N/A", description="Management commentary on performance")
    summary_prospectus: str = Field("N/A", description="Summary prospectus text")
    strategies: str = Field("N/A", description="Investment strategies description")
    risks: str = Field("N/A", description="Principal risks disclosure")
    objective: str = Field("N/A", description="Investment objective")
    
    # Allocation tables
    geographic_allocation: Optional[pd.DataFrame] = Field(None, description="Geographic/country allocation breakdown")
    sector_allocation: Optional[pd.DataFrame] = Field(None, description="Sector allocation breakdown")
    portfolio_composition: Optional[pd.DataFrame] = Field(None, description="Asset type composition")
    industry_allocation: Optional[pd.DataFrame] = Field(None, description="Industry allocation breakdown")
    maturity_allocation: Optional[pd.DataFrame] = Field(None, description="Bond maturity allocation (for fixed income)")
    credit_rating: Optional[pd.DataFrame] = Field(None, description="Credit rating distribution (for fixed income)")
    issuer_allocation: Optional[pd.DataFrame] = Field(None, description="Top issuer allocation")
    
    # Holdings
    top_holdings: Optional[pd.DataFrame] = Field(None, description="Top portfolio holdings")
    derivatives: Optional[Derivatives] = Field(None, description="Derivative positions data")
    non_derivatives: Optional[NonDerivatives] = Field(None, description="Non-derivative holdings data")


# Helper functions for backward compatibility
def fund_name_exists(fund_list: List[FundData], name: str) -> bool:
    """Check if a fund name exists in the list."""
    return any(fund.name == name for fund in fund_list)


def get_fund_by_name(fund_list: List[FundData], name: str) -> Optional[FundData]:
    """Get a fund by name from the list."""
    for fund in fund_list:
        if fund.name == name:
            return fund
    return None


def build_fund_name_index(fund_list: List[FundData]) -> Dict[str, FundData]:
    """Build a dictionary index of funds by name for fast lookup."""
    return {fund.name: fund for fund in fund_list}
