"""
Schema Slicing for Query Classification Categories

Maps query classification categories to focused Neo4j schema subsets.
This reduces token usage and improves LLM focus by only providing relevant
schema portions for each query type.

Categories and their schema slices:
- FUND_BASIC: Fund performance metrics, returns, expense ratios
- FUND_PORTFOLIO: Holdings, allocations, portfolio composition
- FUND_PROFILE: Prospectus content (vector search path)
- COMPANY_FILING: 10-K financials and filing sections
- COMPANY_PEOPLE: Executives, compensation, insider transactions
- NOT_RELATED: No retrieval needed
"""


SCHEMA_SLICES = {
    "not_related": None,  # No schema needed for out-of-scope queries

    "fund_basic": """
# FUND BASIC PERFORMANCE & METRICS
# Scope: expense ratios, annualized returns, net assets, costs, financial highlights

(:Fund {
    ticker,              # Symbol like 'VTI'
    name,                # Full fund name
    netAssets,           # Net assets value
    numberHoldings,      # Total holdings count
    expenseRatio,        # Expense ratio (%)
    advisoryFees,        # Advisory fees (%)
    costsPer10k          # Costs per $10,000 invested
})

# Financial Highlights (fund performance over time)
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {
    year,                # Year of the highlight (on relationship)
    turnover,            # Portfolio turnover (%)
    expenseRatio,        # Expense ratio (%)
    totalReturn,         # Total return (%)
    netAssets,           # Net assets at end of period
    netIncomeRatio       # Net income ratio
})

# Average Returns (multi-year performance)
(:Fund)-[r:HAS_AVERAGE_RETURNS {date}]->(:AverageReturns {
    return1y,            # 1-year annualized return
    return5y,            # 5-year annualized return
    return10y,           # 10-year annualized return
    returnInception      # Since-inception annualized return
})

# Fund Provider context (optional, for fund family info)
(:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund)
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name})
""",

    "fund_portfolio": """
# FUND PORTFOLIO COMPOSITION & ALLOCATIONS
# Scope: holdings, weights, sector/regional breakdowns

(:Fund {
    ticker,              # Symbol like 'VTI'
    name,                # Full fund name
    numberHoldings       # Total holdings count
})

# Portfolio structure
(:Fund)-[:HAS_PORTFOLIO]->(:Portfolio {
    date,                # Portfolio snapshot date
    seriesId             # Series identifier
})

# Individual holdings with weights
(:Portfolio)-[:HAS_HOLDING {
    shares,              # Number of shares
    marketValue,         # Market value of holding
    weight,              # Weight in portfolio (%)
    fairValueLevel,      # Fair value measurement level
    isRestricted,        # Whether restricted
    payoffProfile        # Payoff profile type
}]->(:Holding {
    name,                # Holding name (company/security)
    ticker,              # Ticker symbol
    isin,                # ISIN identifier
    lei,                 # LEI identifier
    category,            # Asset category
    category_desc,       # Category description
    issuer_category,     # Issuer category
    businessAddress      # Business address
})

# Sector allocations
(:Fund)-[h:HAS_SECTOR_ALLOCATION {weight, date}]->(:Sector {
    name                 # Sector name (Technology, Healthcare, etc.)
})

# Regional/geographic allocations
(:Fund)-[h:HAS_REGION_ALLOCATION {weight, date}]->(:Region {
    name                 # Region name (North America, Asia-Pacific, etc.)
})

# Link holdings to companies
(:Holding {ticker})-[:REPRESENTS]->(:Company {ticker})
""",

    "fund_profile": """
# FUND PROFILE & NARRATIVE CONTENT (VECTOR SEARCH)
# Scope: Investment strategy, risks, objectives, performance commentary
# Note: This category uses vector/semantic search over profile embeddings.
# Pass full schema below for reference, but queries will use embeddings.

(:Fund {
    ticker,              # Symbol like 'VTI'
    name                 # Full fund name
})

# Profile with prospectus document
(:Fund)-[:DEFINED_BY {year}]->(:Profile {
    summaryProspectus    # Full prospectus text content
})

(:Profile)-[:HAS_SECTION]->(:Section:Objective {
    text,                # Objective text content
    embedding            # Vector embedding for semantic search
})

(:Profile)-[:HAS_SECTION]->(:Section:PerformanceCommentary {
    text,                # Performance commentary text
    embedding            # Vector embedding for semantic search
})

(:Profile)-[:HAS_SECTION]->(:Section:RiskFactor {
    text,                # Risk factor text
    title,               # Risk factor title
    embedding            # Vector embedding for semantic search
})

(:Profile)-[:HAS_SECTION]->(:Section:StrategyChunk {
    text,                # Strategy text
    title,               # Strategy title
    embedding            # Vector embedding for semantic search
})

# Document reference
(:Profile)-[:EXTRACTED_FROM]->(:Document {
    accession_number,    # SEC accession number
    url,                 # SEC EDGAR URL
    filing_date,         # Filing date
    reportingDate        # Reporting date
})
""",

    "company_filing": """
# COMPANY 10-K FILING & BUSINESS INFORMATION
# Scope: Business financials, income statements, risk factors, MD&A sections

(:Company {
    ticker,              # Stock ticker
    name,                # Company name
    cik                  # SEC Central Index Key
})

# 10-K filing document
(:Company)-[:HAS_FILING {date}]->(:Filing10K {
    filingDate,          # Date filed
    reportingDate        # Fiscal year end date
})

(:Filing10K)-[:EXTRACTED_FROM]->(:Document {
    accession_number,    # SEC accession number
    url,                 # SEC EDGAR URL
    form,                # Form type (10-K)
    filing_date,         # Filing date
    reporting_date       # Reporting period
})

# Business sections with embeddings
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {
    text,                # Business description text
    embedding            # Vector embedding
})

(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {
    text,                # Risk factor text
    embedding            # Vector embedding
})

(:Filing10K)-[:HAS_SECTION]->(:Section:ManagementDiscussion {
    text,                # MD&A text content
    embedding            # Vector embedding
})

(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {
    text,                # Legal proceedings text
    embedding            # Vector embedding
})

# Financial data
(:Filing10K)-[:HAS_SECTION]->(:Section:Financials {
    incomeStatement,     # Income statement JSON
    balanceSheet,        # Balance sheet JSON
    cashFlow,            # Cash flow statement JSON
    fiscalYear           # Fiscal year
})

(:Section:Financials)-[:HAS_METRIC]->(:FinancialMetric {
    label,               # Metric name (Revenue, NetIncome, etc.)
    value                # Metric value
})

(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment {
    label,               # Segment name
    value,               # Segment value
    percentage           # Percentage of total
})

# Properties section
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {
    fullText             # Property details text (note: uses fullText not text)
})
""",

    "company_people": """
# COMPANY PEOPLE, COMPENSATION & INSIDER TRANSACTIONS
# Scope: Executives, CEOs, compensation packages, insider trades

(:Company {
    ticker,              # Stock ticker
    name,                # Company name
    cik                  # SEC Central Index Key
})

# CEO and executives
(:Company)-[r:HAS_CEO {
    ceoCompensation,     # Total compensation
    ceoActuallyPaid,     # Actually paid amount
    date                 # Date of appointment
}]->(:Person {
    name                 # Executive name
})

# Compensation packages
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage {
    totalCompensation,   # Total compensation value
    shareholderReturn,   # Shareholder return metric
    date                 # Compensation year
})

(:CompensationPackage)-[:AWARDED_BY]->(:Company)

(:CompensationPackage)-[:DISCLOSED_IN]->(:Document {
    accession_number,    # SEC accession number
    filing_date,         # Filing date
    url                  # Document URL
})

# Insider transactions
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction {
    transactionDate,     # Date of transaction
    position,            # Person's position (CEO, Officer, etc.)
    transactionType,     # Buy/Sell/Grant/etc.
    shares,              # Number of shares
    price,               # Price per share
    value,               # Total transaction value
    remainingShares      # Shares held after transaction
})

(:InsiderTransaction)-[:MADE_BY]->(:Person {
    name                 # Person name
})

(:InsiderTransaction)-[:EXTRACTED_FROM]->(:Document {
    accession_number,    # SEC accession number
    filing_date,         # Filing date
    url                  # Document URL
})
""",
}


def get_schema_for_category(category: str) -> str:
    """
    Get the schema slice for a given query category.

    Args:
        category: Query classification category (e.g., 'fund_basic', 'company_filing')

    Returns:
        Schema string for that category, or empty string for not_related
    """
    schema = SCHEMA_SLICES.get(category, "")
    return schema if schema else ""


def get_merged_schema(categories: list) -> str:
    """
    Merge multiple category schemas into a single schema string.

    Used when a query is classified into multiple categories and needs
    a union of their schema slices.

    Args:
        categories: List of category names

    Returns:
        Merged schema string containing all relevant nodes and relationships
    """
    merged_parts = []

    for category in categories:
        if category == "not_related":
            continue

        schema = SCHEMA_SLICES.get(category, "")
        if schema:
            merged_parts.append(f"# {category.upper()}\n{schema}")

    return "\n".join(merged_parts)


# Summary of schema sizes (approximate token counts)
SCHEMA_TOKEN_COUNTS = {
    "not_related": 0,
    "fund_basic": 150,
    "fund_portfolio": 200,
    "fund_profile": 180,
    "company_filing": 250,
    "company_people": 200,
}


def estimate_tokens(categories: list) -> int:
    """Estimate token count for merged schema."""
    return sum(SCHEMA_TOKEN_COUNTS.get(cat, 0) for cat in categories)
