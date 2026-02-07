from pydantic import BaseModel, Field, model_validator
from datetime import date, datetime
from typing import Optional, List, Dict

class FilingMetadata(BaseModel):
    """The 'Source' - Information about the specific SEC document."""
    accession_number: str = Field(..., description="SEC unique identifier (e.g., 0000320193-24-000123)")
    filing_type: str = Field(..., description="Type of filing: 10-K, 10-Q, 8-K, etc.")
    filing_date: date = Field(..., description="Date the document was submitted to EDGAR")
    report_period_end: date = Field(..., description="The end date of the financial period being reported")
    filing_url: str = Field(..., description="Direct link to the HTML/iXBRL document")
    cik: str = Field(..., description="Central Index Key of the filer")


class FinancialSegment(BaseModel):
    """Represents a breakdown of a main metric (e.g., 'iPhone' for Revenue)."""
    label: str = Field(..., description="The name of the segment (e.g., 'iPhone', 'China')")
    amount: float = Field(..., description="The value of this segment")
    axis: Optional[str] = Field(None, description="The dimension axis (e.g., 'Product', 'Geography')")
    percentage: Optional[float] = Field(None, description="Percentage contribution to the total (0-100)")

class FinancialMetric(BaseModel):
    """Holds the total value AND its segments together."""
    value: Optional[float] = Field(None, description="The total aggregate value")
    segments: List[FinancialSegment] = Field(default_factory=list, description="Breakdowns of this total")
    label: Optional[str] = Field(None, description="The name of this metric (e.g., 'Revenue', 'Cost of Revenue')")
    
    @model_validator(mode='after')
    def compute_segment_weights(self):
        """
        After the model is initialized, iterate through segments
        and calculate their percentage relative to the total value.
        """
        # 1. We need a total value > 0 to divide
        total_val = self.value
        
        if total_val and total_val > 0 and self.segments:
            for segment in self.segments:
                # Calculate %: (Segment Amount / Total Amount) * 100
                calculated_pct = (segment.amount / total_val) * 100
                
                # Round to 2 decimals for cleanliness
                segment.percentage = round(calculated_pct, 2)
        return self

class IncomeStatement(BaseModel):
    """The 'Income Statement' - Financial performance metrics with multi-year support."""
    
    # Fiscal period identification
    fiscal_year: Optional[int] = Field(None, description="The fiscal year for this income statement (e.g., 2024)")
    period_end_date: Optional[date] = Field(None, description="The end date of the reporting period")
    
    # Cost of sales
    cost_of_sales: Optional[FinancialMetric] = Field(None, description="Cost of goods sold (COGS)")

    # Core Financial Metrics (using FinancialMetric to support segments)
    revenue: Optional[FinancialMetric] = Field(None, description="Total revenue with product/geography breakdowns")
    gross_profit: Optional[FinancialMetric] = Field(None, description="Revenue minus COGS")
    
    # Operating Expenses
    operating_expenses: Optional[FinancialMetric] = Field(None, description="Total operating expenses")
    
    # Operating Results
    operating_income: Optional[FinancialMetric] = Field(None, description="Income from operations (EBIT)")
    
    # Other Income Expenses
    other_income_expense: Optional[FinancialMetric] = Field(None, description="Other income and expenses")

    # Pre-tax and Tax
    pretax_income: Optional[FinancialMetric] = Field(None, description="Total profit before taxes")
    provision_for_income_taxes: Optional[FinancialMetric] = Field(None, description="Tax expense for the period")
    
    # Net Income
    net_income: Optional[FinancialMetric] = Field(None, description="Bottom line profit after all expenses and taxes")
    
    # Per Share Metrics
    basic_earnings_per_share: Optional[float] = Field(None, description="EPS using actual shares outstanding")
    diluted_earnings_per_share: Optional[float] = Field(None, description="EPS including potential dilution from options/convertibles")
    
    # Number of shares
    basic_shares_outstanding: Optional[float] = Field(None, description="Number of shares outstanding")
    diluted_shares_outstanding: Optional[float] = Field(None, description="Number of shares outstanding including potential dilution from options/convertibles")
    
    # Computed Profitability Ratios
    @property
    def gross_margin_percent(self) -> Optional[float]:
        """Gross Margin % = (Gross Profit / Revenue) × 100"""
        if self.revenue and self.gross_profit and self.revenue.value and self.gross_profit.value and self.revenue.value > 0:
            return (self.gross_profit.value / self.revenue.value) * 100
        return None
    
    @property
    def operating_margin_percent(self) -> Optional[float]:
        """Operating Margin % = (Operating Income / Revenue) × 100"""
        if self.revenue and self.operating_income and self.revenue.value and self.operating_income.value and self.revenue.value > 0:
            return (self.operating_income.value / self.revenue.value) * 100
        return None
    
    @property
    def net_profit_margin_percent(self) -> Optional[float]:
        """Net Profit Margin % = (Net Income / Revenue) × 100"""
        if self.revenue and self.net_income and self.revenue.value and self.net_income.value and self.revenue.value > 0:
            return (self.net_income.value / self.revenue.value) * 100
        return None
    
    @property
    def effective_tax_rate_percent(self) -> Optional[float]:
        """Effective Tax Rate = (Provision for Taxes / Pretax Income) × 100"""
        if self.pretax_income and self.provision_for_income_taxes and self.pretax_income.value and self.provision_for_income_taxes.value and self.pretax_income.value > 0:
            return (self.provision_for_income_taxes.value / self.pretax_income.value) * 100
        return None

class ExecutiveCompensation(BaseModel):
    """Represents the executive compensation section of a DEF 14A filing."""
    url: Optional[str] = Field(None, description="The URL of the executive compensation section")
    form: Optional[str] = Field(None, description="The form type of the executive compensation section")
    text: Optional[str] = Field(None, description="The text of the executive compensation section")
    ceo_name: Optional[str] = Field(None, description="The name of the CEO")
    ceo_compensation: Optional[float] = Field(None, description="The compensation of the CEO")
    ceo_actually_paid: Optional[float] = Field(None, description="The actually paid of the CEO")
    shareholder_return: Optional[float] = Field(None, description="The shareholder return of the CEO")


class Filing10K(BaseModel):
    """Represents a single 10-K filing with all its associated data."""
    
    # Filing Metadata
    filing_metadata: FilingMetadata = Field(..., description="Metadata about the filing")
    
    # Business Information
    business_information: Optional[str] = Field(None, description="The business information section of the filing")
    management_discussion_and_analysis: Optional[str] = Field(None, description="The management discussion and analysis section of the filing")
    legal_proceedings: Optional[str] = Field(None, description="The legal proceedings section of the filing")
    properties: Optional[str] = Field(None, description="The properties section of the filing")
    
    # Income Statements (multiple periods can be in one filing)
    income_statements: Dict[date, IncomeStatement] = Field(default_factory=dict, description="Income statements for multiple fiscal periods in this filing")
    
    # Raw text content
    income_statement_text: Optional[str] = Field(None, description="The whole content of the income statement in text form")
    balance_sheet_text: Optional[str] = Field(None, description="The whole content of the balance sheet in text form")
    cash_flow_text: Optional[str] = Field(None, description="The whole content of the cash flow in text form")
    
    # Risk Factors
    risk_factors: Optional[str] = Field(None, description="The risk factors section of the filing")
    
    def get_income_statement(self, period_end_date: date) -> Optional[IncomeStatement]:
        """Get income statement for a specific period end date."""
        return self.income_statements.get(period_end_date)
    
    def get_latest_income_statement(self) -> Optional[IncomeStatement]:
        """Get the most recent income statement in this filing."""
        if not self.income_statements:
            return None
        return self.income_statements[max(self.income_statements.keys())]

class InsiderTransaction(BaseModel):
    date: Optional[str] = None
    insider_name: Optional[str] = None
    position: Optional[str] = None
    transaction_type: Optional[str] = None  # "BUY", "SELL", "GRANT", "VESTING", "TAX"
    shares: Optional[int] = None
    security_type: Optional[str] = None
    price: Optional[float] = None
    value: Optional[float] = None
    remaining_shares: Optional[int] = None
    filing_url: Optional[str] = None
    form: Optional[str] = None

class CompanyEntity(BaseModel):
    """The 'Entity' - Static identification for the company."""
    name: str = Field(..., description="Legal name of the company")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    cik: Optional[str] = Field(None, description="Unique SEC CIK number")
    
    # 10-K Filings keyed by filing date
    filings_10k: Dict[date, Filing10K] = Field(default_factory=dict, description="10-K filings for this company")
    
    # Executive Compensation
    executive_compensation: Optional[ExecutiveCompensation] = Field(None, description="The executive compensation section of the filing")
    
    # Insider Trades
    insider_trades: List[InsiderTransaction] = Field(default_factory=list)
    
    def get_filing(self, filing_date: date) -> Optional[Filing10K]:
        """Get a specific 10-K filing by filing date."""
        return self.filings_10k.get(filing_date)
    
    def get_latest_filing(self) -> Optional[Filing10K]:
        """Get the most recent 10-K filing."""
        if not self.filings_10k:
            return None
        return self.filings_10k[max(self.filings_10k.keys())]
    
    def get_all_income_statements(self) -> List[IncomeStatement]:
        """Get all income statements across all filings."""
        statements = []
        for filing in self.filings_10k.values():
            statements.extend(filing.income_statements.values())
        return sorted(statements, key=lambda x: x.period_end_date or date.min, reverse=True)
    
    def pretty_print(self) -> str:
        """
        Generate a formatted string representation of the company and all its financial data.
        Returns a nicely indented, human-readable view of all filings and income statements.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"COMPANY: {self.name}")
        lines.append("=" * 80)
        
        if self.ticker:
            lines.append(f"Ticker: {self.ticker}")
        if self.cik:
            lines.append(f"CIK: {self.cik}")
        
        lines.append("")
        lines.append(f"Total 10-K Filings: {len(self.filings_10k)}")
        lines.append("")
        
        # Sort filings by filing date (most recent first)
        sorted_filings = sorted(
            self.filings_10k.items(), 
            key=lambda x: x[0], 
            reverse=True
        )
        
        for filing_date, filing in sorted_filings:
            lines.append("=" * 80)
            lines.append(f"10-K FILING - Filed: {filing_date}")
            lines.append("=" * 80)
            lines.append(f"  Accession #: {filing.filing_metadata.accession_number}")
            lines.append(f"  Filing Type: {filing.filing_metadata.filing_type}")
            lines.append(f"  Report Period End: {filing.filing_metadata.report_period_end}")
            lines.append("")
            
            # Sort income statements within this filing by period date (most recent first)
            sorted_statements = sorted(
                filing.income_statements.items(),
                key=lambda x: x[0],
                reverse=True
            )
            
            for period_date, stmt in sorted_statements:
                lines.append("-" * 80)
                lines.append(f"INCOME STATEMENT - Period Ending: {period_date}")
                if stmt.fiscal_year:
                    lines.append(f"Fiscal Year: {stmt.fiscal_year}")
                lines.append("-" * 80)
                
                # Financial metrics
                lines.append("  Financial Metrics:")
                lines.append("")
                
                # Helper function to format a financial metric
                def format_metric(label: str, metric: Optional[FinancialMetric], indent: str = "    ") -> List[str]:
                    metric_lines = []
                    if metric and metric.value is not None:
                        # Use the metric's label if available, otherwise use the provided label
                        display_label = f"{label} ({metric.label})" if metric.label else label
                        metric_lines.append(f"{indent}{display_label}: ${metric.value:,.2f}")
                        
                        # Show segments if they exist, grouped by axis
                        if metric.segments:
                            # Group segments by axis
                            segments_by_axis = {}
                            for seg in metric.segments:
                                axis = seg.axis or "Other"
                                if axis not in segments_by_axis:
                                    segments_by_axis[axis] = []
                                segments_by_axis[axis].append(seg)
                            
                            # Display each axis group
                            metric_lines.append(f"{indent}  Segments:")
                            for axis, segments in segments_by_axis.items():
                                metric_lines.append(f"{indent}    [{axis}]")
                                for seg in segments:
                                    pct_str = f" ({seg.percentage:.2f}%)" if seg.percentage else ""
                                    metric_lines.append(f"{indent}      • {seg.label}: ${seg.amount:,.2f}{pct_str}")
                    return metric_lines
            
                # Revenue section
                if stmt.revenue:
                    lines.extend(format_metric("Revenue", stmt.revenue))
                    lines.append("")
                
                if stmt.cost_of_sales:
                    lines.extend(format_metric("Cost of Sales", stmt.cost_of_sales))
                
                if stmt.gross_profit:
                    lines.extend(format_metric("Gross Profit", stmt.gross_profit))
                    if stmt.gross_margin_percent:
                        lines.append(f"      → Gross Margin: {stmt.gross_margin_percent:.2f}%")
                    lines.append("")
                
                # Operating expenses
                lines.append("    Operating Expenses:")
               
                if stmt.operating_expenses:
                    lines.extend(format_metric("Total Operating Expenses", stmt.operating_expenses, "      "))
                lines.append("")
                
                # Operating income
                if stmt.operating_income:
                    lines.extend(format_metric("Operating Income", stmt.operating_income))
                    if stmt.operating_margin_percent:
                        lines.append(f"      → Operating Margin: {stmt.operating_margin_percent:.2f}%")
                    lines.append("")
                
                # Pre-tax and taxes
                if stmt.pretax_income:
                    lines.extend(format_metric("Pre-tax Income", stmt.pretax_income))
                
                if stmt.provision_for_income_taxes:
                    lines.extend(format_metric("Income Taxes", stmt.provision_for_income_taxes))
                    if stmt.effective_tax_rate_percent:
                        lines.append(f"      → Effective Tax Rate: {stmt.effective_tax_rate_percent:.2f}%")
                    lines.append("")
                
                # Net income
                if stmt.net_income:
                    lines.extend(format_metric("Net Income", stmt.net_income))
                    if stmt.net_profit_margin_percent:
                        lines.append(f"      → Net Profit Margin: {stmt.net_profit_margin_percent:.2f}%")
                    lines.append("")
                
                # EPS
                if stmt.basic_earnings_per_share or stmt.diluted_earnings_per_share:
                    lines.append("    Earnings Per Share:")
                    if stmt.basic_earnings_per_share:
                        lines.append(f"      Basic EPS: ${stmt.basic_earnings_per_share:.2f}")
                    if stmt.diluted_earnings_per_share:
                        lines.append(f"      Diluted EPS: ${stmt.diluted_earnings_per_share:.2f}")
                    lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)