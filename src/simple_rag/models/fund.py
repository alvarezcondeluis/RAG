from typing import Optional, Dict, List
from decimal import Decimal
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum
import re
from datetime import date, datetime
from decimal import Decimal


class FilingMetadata(BaseModel):
    """Metadata about the SEC filing source for fund data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    accession_number: str = Field(..., description="SEC unique identifier (e.g., 0001193125-24-123456)")
    filing_date: date = Field(..., description="Date the document was submitted to EDGAR")
    reporting_date: date = Field(..., description="The period end date being reported")
    url: str = Field(..., description="Direct link to the filing on EDGAR")
    form: Optional[str] = Field(None, description="Name/title of the document")
    
class ContentChunk(BaseModel):
    """A chunk of content with title, text, and embedding."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = Field(description="Unique identifier for the chunk")
    title: str = Field(description="Title or heading of the chunk")
    text: str = Field(description="Text content of the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")

class ChartData(BaseModel):
    """A chart with title, type, and SVG content."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    title: str = Field(..., description="Title of the chart")
    image_type: str = Field("performance", description="Type (performance, allocation, etc.)")
    svg_content: str = Field(..., description="The raw XML string of the SVG")

class AverageReturnSnapshot(BaseModel):
    """Snapshot of fund returns over different time periods."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    return_1y: Optional[float] = Field(description="1-year return percentage from the relationship date")
    return_5y: Optional[float] = Field(description="5-year return percentage from the relationship date")
    return_10y: Optional[float] = Field(description="10-year return percentage from the relationship date")
    return_inception: Optional[float] = Field(description="Return since fund inception (percentage)")
    
    @field_validator('return_1y', 'return_5y', 'return_10y', 'return_inception', mode='before')
    @classmethod
    def parse_return_fields(cls, v):
        """Convert string return values to floats, handle None properly."""
        if v is None:
            return None
        
        if isinstance(v, (int, float, Decimal)):
            return float(v)
        
        if isinstance(v, str):
           
            if v in ['', 'N/A', 'n/a', 'NA', 'None', 'null', '—', '-']:
                return None
            try:
                return float(v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to convert '{v}' to float: {str(e)}")
        
        raise TypeError(f"Cannot convert type {type(v)} to float")

class PortfolioHolding(BaseModel):
    """Simplified holding for fund analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(description="Security name")
    cusip: Optional[str] = Field(None, description="CUSIP identifier")
    isin: Optional[str] = Field(None, description="ISIN identifier")
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

class AnnualReportGeneralInformation(BaseModel):
    """General information about the annual report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    
class FundData(BaseModel):
    """Complete fund data structure with all fund information and metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Core identifiers
    ncsr_metadata: Optional[FilingMetadata] = Field(None, description="Metadata about the SEC filing source for fund data")
    summary_prospectus_metadata: Optional[FilingMetadata] = Field(None, description="Metadata about the SEC filing source for fund data")
    nport_metadata: Optional[FilingMetadata] = Field(None, description="Metadata about the SEC filing source for fund data")
    name: str = Field(description="Official fund name")
    registrant: str = Field(description="Registered investment company name")
    provider: Optional[str] = Field(None, description="Name of the company that hosts the fund")
    context_id: str = Field(description="SEC EDGAR context identifier")
    share_class: Optional[ShareClassType] = Field(None, description="Share class designation (e.g., Admiral, Investor)")
    ticker: str = Field("N/A", description="Stock ticker symbol")
    security_exchange: Optional[str] = Field(None, description="Primary listing exchange")
    series_id: Optional[str] = Field(None, description="Series identifier")
    
    # Financial metrics
    costs_per_10k: int = Field(0, description="Cost per $10,000 invested")
    expense_ratio: float = Field(0.0, description="Annual expense ratio as percentage (Annual fee charged by the fund)")
    net_assets: float = Field(0.0, description="Total net assets under management in millions")
    turnover_rate: float = Field(0.0, description="percentage of a fund's holdings that have been replaced (bought and sold) over a one-year period")
    advisory_fees: float = Field(0.0, description="Investment advisory fees")
    n_holdings: int = Field(0, description="Total number of holdings in portfolio")
    
    report_date: Optional[date] = Field(None, description="Date of the report or filing")
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
    performance_commentary_embedding: Optional[List[float]] = Field(None, description="Embedding for performance commentary")
    summary_prospectus: str = Field("N/A", description="Summary prospectus text")
    strategies: str = Field("N/A", description="Investment strategies description")
    strategies_chunks: Optional[List[ContentChunk]] = Field(
        default_factory=list,
        description="List of strategy content chunks with embeddings"
    )
    risks: str = Field("N/A", description="Principal risks disclosure")
    risks_chunks: Optional[List[ContentChunk]] = Field(
        default_factory=list,
        description="List of risk content chunks with embeddings"
    )
    objective: str = Field("N/A", description="Investment objective")
    objective_embedding: Optional[List[float]] = Field(None, description="Embedding for investment objective")
    
    # Charts and visualizations
    charts: Optional[List[ChartData]] = Field(
        default_factory=list,
        description="List of charts and visualizations (SVG format)"
    )
    
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


    @field_validator('costs_per_10k', 'n_holdings', mode='before')
    @classmethod
    def parse_integer_fields(cls, v):
        """Convert cleaned numeric values to integers."""
        if isinstance(v, int):
            return v
        
        if isinstance(v, (float, Decimal)):
            return int(v)
        
        if isinstance(v, str):
            v = v.replace('$', '').replace(',', '')
            try:
                return int(float(v))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to convert '{v}' to integer: {str(e)}")
        
        raise TypeError(f"Cannot convert type {type(v)} to integer")
    
    @field_validator('expense_ratio', 'net_assets', 'turnover_rate', 'advisory_fees', 'costs_per_10k', mode='before')
    @classmethod
    def parse_float_fields(cls, v):
        """Convert cleaned numeric values to floats."""
        if isinstance(v, (int, float, Decimal)):
            return float(v)
        
        if isinstance(v, str):
            v = v.replace('$', '').replace(',', '')
            try:
                return float(v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to convert '{v}' to float: {str(e)}")
        
        raise TypeError(f"Cannot convert type {type(v)} to float")
    

    @field_validator('report_date', mode='before')
    @classmethod
    def parse_date_field(cls, v):
        """Convert string date to datetime.date object."""
        if v is None or (isinstance(v, str) and v.strip() in ['N/A', 'n/a', 'NA', 'None', 'null', '—', '-', '']):
            return None
        
        if isinstance(v, date):
            return v
        
        if isinstance(v, datetime):
            return v.date()
        
        if isinstance(v, str):
            # Try common date formats
            date_formats = [
                '%Y-%m-%d',      # 2024-01-15
                '%m/%d/%Y',      # 01/15/2024
                '%d/%m/%Y',      # 15/01/2024
                '%Y/%m/%d',      # 2024/01/15
                '%B %d, %Y',     # January 15, 2024
                '%b %d, %Y',     # Jan 15, 2024
                '%Y%m%d',        # 20240115
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(v.strip(), fmt).date()
                except ValueError:
                    continue
            
            raise ValueError(f"Could not parse date from '{v}'. Expected formats: YYYY-MM-DD, MM/DD/YYYY, etc.")
        
        raise TypeError(f"Cannot convert type {type(v)} to date")
    
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
