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

(:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund {
    ticker,              # Symbol like 'VTI'
    name,                # Full fund name
    netAssets,           # Net assets value
    numberHoldings,      # Total holdings count
    expenseRatio,        # Expense ratio (%)
    advisoryFees,        # Advisory fees (%)
    costsPer10k          # Costs per $10,000 invested
})
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
(:Fund)-[:EXTRACTED_FROM]->(:Document {url, type, filingDate, accessionNumber})

# Financial Highlights — year is on the RELATIONSHIP, not the node
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {
    turnover,            # Portfolio turnover (%)
    expenseRatio,        # Expense ratio (%)
    totalReturn,         # Total return (%)
    netAssets,           # Net assets at end of period
    netAssetsValueBeginning,
    netAssetsValueEnd,
    netIncomeRatio       # Net income ratio
})

# Average Returns (multi-year performance)
(:Fund)-[r:HAS_AVERAGE_RETURNS {date}]->(:AverageReturns {
    return1y,            # 1-year annualized return
    return5y,            # 5-year annualized return
    return10y,           # 10-year annualized return
    returnInception      # Since-inception annualized return
})

QUERY RULES:
- For name searches use CALL db.index.fulltext.queryNodes('fundNameIndex', 'search_term')
- For exact ticker matching use {ticker: 'VTI'} directly.
- numberHoldings is pre-calculated on the Fund node — do NOT count holdings manually.
- The year is on the RELATIONSHIP [r:HAS_FINANCIAL_HIGHLIGHT {year}], not the node: use r.year.
- turnover is absolute (2 = 2%, not 0.02).
- ALWAYS return the source of the information via the EXTRACTED_FROM document node.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. STRICT SCHEMA ALIGNMENT: `netAssets`, `expenseRatio`, and `advisoryFees` are on `FinancialHighlight`, NOT on `Fund`. Access them via the HAS_FINANCIAL_HIGHLIGHT relationship.
2. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
3. COMPARING ENTITIES: When asked to compare multiple funds (e.g., "Compare VTI and VOO"), return each fund as a separate row using `IN` (e.g., `WHERE f.ticker IN ['VTI', 'VOO'] RETURN f.ticker...`).
4. LATEST / LAST DATA: When a user asks for the "last", "latest", or "current" metric, ALWAYS use `ORDER BY r.year DESC LIMIT 1`.
5. HISTORICAL GROWTH (Since X years ago): MATCH the highlights, order by year, `collect(fh)`, and compare `highlights[0]` (latest) to `highlights[X]` (previous).
6. DIVISION BY ZERO: When calculating percentages, ALWAYS use `CASE WHEN denominator = 0 THEN 0 ELSE (numerator * 100.0 / denominator) END`.
7. INCOMPLETE FILTERS: NEVER generate empty or incomplete property filters like `(n:Label {name})`. Only include explicit values like `{name: 'Vanguard'}`.
""",

    "fund_portfolio": """
# FUND PORTFOLIO COMPOSITION & ALLOCATIONS
# Scope: holdings, weights, sector/regional breakdowns, asset categories

(:Fund {
    ticker,              # Symbol like 'VTI'
    name,                # Full fund name
    numberHoldings       # Total holdings count (pre-calculated — use this, do NOT count)
})

# Portfolio structure
(:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio {
    date,                # Portfolio snapshot date
    seriesId,            # Series identifier
    count                # Number of holdings
})

# Individual holdings with weights (weight, marketValue, etc. are on the RELATIONSHIP)
(p)-[r:HAS_HOLDING {
    shares,              # Number of shares
    marketValue,         # Market value of holding
    weight,              # Weight in portfolio (%)
    fairValueLevel,      # Fair value measurement level
    isRestricted,        # Whether restricted
    payoffProfile        # Payoff profile type
}]->(h:Holding {
    name,                # Holding name (company/security)
    ticker,              # Ticker symbol
    isin,                # ISIN identifier
    lei,                 # LEI identifier
    category,            # Asset category
    category_desc,       # Category description
    issuer_category,     # Issuer category
    businessAddress      # Business address
})

# Asset type classification
(h)-[:OF_ASSET_TYPE]->(:AssetCategory {
    code,                # Asset category code
    name,                # Asset category name
    type,                # Broad asset type
    subtype              # Specific subtype
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
(h)-[:REPRESENTS]->(:Company {ticker, name})

(p)-[:EXTRACTED_FROM]->(:Document {url, type, filingDate, reportingDate, accessionNumber})

QUERY RULES:
- Use p.count for number of holdings — do NOT count holdings manually with COUNT(h).
- weight, marketValue, payoffProfile are on the [r:HAS_HOLDING] RELATIONSHIP, not the Holding node.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. STRICT SCHEMA ALIGNMENT: `payoffProfile`, `marketValue`, and `weight` are properties of the `[r:HAS_HOLDING]` relationship, NOT the `Holding` node.
2. INLINE MATH: NEVER use math operators (`>`, `<`) inside curly braces in MATCH patterns (BAD: `[r:HAS_HOLDING {marketValue: > 10000}]`). ALWAYS use a `WHERE` clause (GOOD: `WHERE r.marketValue > 10000`).
3. DIVISION BY ZERO: When calculating percentage ratios, ALWAYS use `CASE WHEN total = 0 THEN 0 ELSE (part * 100.0 / total) END`.
4. COMMAS: Ensure proper commas are placed between variables in the `RETURN` statement before aggregate functions (e.g., `RETURN a, COUNT(b)`).
""",

    "fund_profile": """
# FUND PROFILE & NARRATIVE CONTENT
# Scope: Investment strategy, risks, objectives, performance commentary
# Note: vector search lives on :Chunk nodes (not :Section). Objectives embed on the Section itself.

(:Fund {
    ticker,              # Symbol like 'VTI'
    name                 # Full fund name
})

# Profile with prospectus document
(:Fund)-[:DEFINED_BY {year}]->(:Profile {
    summaryProspectus    # Full prospectus text content
})

# === SECTIONS ===
# Embeddings live on Section:Objective AND on child :Chunk nodes for Strategy/RiskFactor.
# PerformanceCommentary has NO embedding — structural retrieval only.

(:Profile)-[:HAS_SECTION]->(:Section:Objective {
    text,                # Objective text
    title,               # Objective title
    embedding            # Vector embedding LIVES ON THE SECTION (no chunks)
})

(:Profile)-[:HAS_SECTION]->(:Section:PerformanceCommentary {
    text,                # Performance commentary text
    title                # No embedding available
})

(:Profile)-[:HAS_SECTION]->(:Section:Strategy {
    text,                # Strategy section text
    title                # Embeddings are on child :Chunk nodes
})

(:Profile)-[:HAS_SECTION]->(:Section:RiskFactor {
    text,                # Risk factor text
    title                # Embeddings are on child :Chunk nodes
})

# === CHUNKS (the actual embedded units) ===
(:Section:Strategy)-[:HAS_CHUNK]->(:Chunk {
    text,                # Chunk text (smaller-grained excerpt)
    embedding            # Vector embedding for semantic search
})

(:Section:RiskFactor)-[:HAS_CHUNK]->(:Chunk {
    text,                # Chunk text
    embedding            # Vector embedding for semantic search
})

# === VECTOR INDEXES ===
# CALL db.index.vector.queryNodes('profileChunkIndex', $k, $queryVector)
#   → returns :Chunk nodes from Profile Strategy / RiskFactor sections.
#   Traverse: chunk <-[:HAS_CHUNK]-(section:Section) <-[:HAS_SECTION]-(p:Profile) <-[:DEFINED_BY]-(f:Fund)
#   Filter by section label, e.g.  WHERE 'Strategy' IN labels(section)
#
# CALL db.index.vector.queryNodes('profileObjectiveIndex', $k, $queryVector)
#   → returns :Section:Objective nodes (embedding ON the section).
#   Traverse: objective <-[:HAS_SECTION]-(p:Profile) <-[:DEFINED_BY]-(f:Fund)

(:Profile)-[:EXTRACTED_FROM]->(:Document {
    accession_number,    # SEC accession number
    url,                 # SEC EDGAR URL
    filing_date,         # Filing date
    reportingDate        # Reporting date
})

QUERY RULES:
- These nodes have embeddings — prefer vector search for semantic queries.
- ALWAYS return ticker and name alongside the requested data.
- Do NOT attempt to extract netAssets or numerical performance from these text nodes.
""",

    "company_filing": """
# COMPANY 10-K FILING & BUSINESS INFORMATION
# Scope: Business financials, income statements, risk factors, MD&A sections.
# Embeddings live on :Chunk nodes (NOT on Section nodes). Use filing10kChunkIndex.

(:Company {
    ticker,              # Stock ticker like 'AAPL', 'MSFT'
    name,                # Company name
    cik                  # SEC Central Index Key
})

# 10-K filing — year is on the REPORTS_IN relationship
(:Company)-[:REPORTS_IN {year}]->(:Filing10K)

(:Filing10K)-[:EXTRACTED_FROM]->(:Document {
    accession_number,    # SEC accession number
    url,                 # SEC EDGAR URL
    form,                # Form type (10-K)
    filing_date,         # Filing date
    reporting_date       # Reporting period
})

# === SECTIONS (text + title only — NO embeddings here) ===
# IMPORTANT: 'ManagemetDiscussion' is the actual label in the DB (typo preserved).
(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:ManagemetDiscussion {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {text, title})

# === CHUNKS (the embedded units — used by filing10kChunkIndex) ===
(:Section:RiskFactor)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:BusinessInformation)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:LegalProceeding)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:ManagemetDiscussion)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:Properties)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})

# === VECTOR INDEX ===
# CALL db.index.vector.queryNodes('filing10kChunkIndex', $k, $queryVector)
#   → returns :Chunk nodes from any 10-K Section.
#   Traverse: chunk <-[:HAS_CHUNK]-(section:Section) <-[:HAS_SECTION]-(filing:Filing10K)
#             <-[:REPORTS_IN]-(c:Company)
#   Filter by section label, e.g.  WHERE 'RiskFactor' IN labels(section)

# === FINANCIAL DATA (structured, no embeddings) ===
(:Filing10K)-[:HAS_FINANCIALS]->(:Section:Financials {
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

QUERY RULES:
- Company ticker is a stock ticker like 'AAPL', 'MSFT'.
- The MD&A section label is 'ManagemetDiscussion' (one 't' — typo preserved from source data).
- The financials relationship is HAS_FINANCIALS (also exists as HAS_FINACIALS with one 'N' — use HAS_FINANCIALS).

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
2. LATEST FILING: When asked for the "latest" or "newest" filing/metrics, use `ORDER BY r.year DESC LIMIT 1` on the REPORTS_IN relationship.
""",

    "company_people": """
# COMPANY PEOPLE, COMPENSATION & INSIDER TRANSACTIONS
# Scope: Fund managers, company executives, compensation packages, insider trades

# Fund managers
(:Fund {ticker, name})-[:MANAGED_BY {year}]->(:Person {name})

(:Company {
    ticker,              # Stock ticker
    name,                # Company name
    cik                  # SEC Central Index Key
})

# CEO and executives
(:Company)-[r:HAS_CEO {
    ceoCompensation,     # Total compensation
    ceoActuallyPaid,     # Actually paid amount
    date                 # Year/date
}]->(:Person {name})

# Compensation packages
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage {
    totalCompensation,   # Total compensation value
    shareholderReturn,   # Shareholder return metric
    date                 # Compensation year
})
(:CompensationPackage)-[:AWARDED_BY]->(:Company)
(:CompensationPackage)-[:DISCLOSED_IN]->(:Document {
    accession_number, filing_date, url
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
(:InsiderTransaction)-[:MADE_BY]->(:Person {name})
(:InsiderTransaction)-[:EXTRACTED_FROM]->(:Document {
    accession_number, filing_date, url
})

QUERY RULES:
- Fund managers are linked via MANAGED_BY (on Fund, not Company).
- Company CEOs and executives are linked via HAS_CEO (on Company).
- Use personNameIndex for fuzzy person name search: CALL db.index.fulltext.queryNodes('personNameIndex', 'name')

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. WHERE CLAUSE POSITION: `WHERE` must immediately follow `MATCH` or `WITH`. NEVER place `WHERE` after `RETURN`.
2. INLINE MATH: NEVER use math operators (`>`, `<`) inside node or relationship patterns. Use `WHERE` clauses.
""",
}


def get_schema_for_category(category: str) -> str:
    """Get the schema slice for a given query category, or empty string for not_related."""
    schema = SCHEMA_SLICES.get(category, "")
    return schema if schema else ""


def get_merged_schema(categories: list) -> str:
    """
    Merge multiple category schemas into a single schema string.

    Used when a query is classified into multiple categories and needs
    a union of their schema slices.
    """
    parts = []
    for category in categories:
        if category == "not_related":
            continue
        schema = SCHEMA_SLICES.get(category, "")
        if schema:
            parts.append(f"=== Schema: {category} ===\n{schema}")
    return "\n\n".join(parts)
