"""
Neo4j Schema Definitions for Text2Cypher Translation.

This module contains the complete Neo4j graph schema used for generating
Cypher queries from natural language. The schema is extracted from the
main text2cypher module for better organization and maintainability.
"""


DETAILED_SCHEMA = """
# ============================================================
# === FUND MANAGEMENT STRUCTURE ===
# ============================================================
(:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund)
# === FUND NODE PROPERTIES (Use these directly!) ===
(:Fund {
    ticker,              # Symbol like 'VTI' (use for matching symbols)
    name,                # Full name like 'Vanguard Total Stock Market Index Fund'
    securityExchange,    # Exchange like 'NASDAQ', 'NYSE'
    costsPer10k,         # Costs per $10,000 invested (numeric)
    advisoryFees,        # Advisory fees (numeric)
    numberHoldings,      # Total number of holdings (integer) - USE THIS, not count(h)!
    expenseRatio,        # Expense ratio (numeric)
    netAssets,           # Net assets value (numeric) - DIRECTLY ON FUND NODE
    turnoverRate        # Portfolio Turnover Rate in ABSOLUTE terms (e.g., 2 means 2%, NOT 0.02)
})
# FULL-TEXT INDEXES (use for fuzzy/partial name matching):
# - Provider.name -> use CALL db.index.fulltext.queryNodes('providerNameIndex', 'search_term')
# - Trust.name -> use CALL db.index.fulltext.queryNodes('trustNameIndex', 'search_term')
# - Fund.name -> use CALL db.index.fulltext.queryNodes('fundNameIndex', 'search_term')
# - Person.name -> use CALL db.index.fulltext.queryNodes('personNameIndex', 'search_term')
# For exact ticker matching, use {ticker: 'VTI'} (no index needed)

# === FUND RELATIONSHIPS ===
# Share Classes
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
# Document node — NOTE: property names use exact DB casing (accessionNumber, filingDate)
(:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})
# Profile (versioned by date)
(:Fund)-[:DEFINED_BY {date}]->(:Profile {summaryProspectus})
(:Profile)-[:HAS_OBJECTIVE_CHUNK]->(:Objective {id, text, embedding})
(:Profile)-[:HAS_PERFORMANCE_COMMENTARY_CHUNK]->(:PerformanceCommentary {id, text, embedding})
(:Profile)-[:HAS_RISK_CHUNK]->(:RiskChunk {id, title, text, embedding})
(:Profile)-[:HAS_STRATEGY_CHUNK]->(:StrategyChunk {id, title, text, embedding})
(:Profile)-[:EXTRACTED_FROM]->(:Document)
# Charts/Images
(:Fund)-[:HAS_CHART]->(:Image {category, svg, title})

# Management Team (date on the relationship)
(:Fund)-[:MANAGED_BY {date}]->(:Person {name})

# Average Returns (versioned by date)
(:Fund)-[:HAS_AVERAGE_RETURNS {date}]->(:AverageReturns {return1y, return5y, return10y, returnInception})

# Allocations (by report date)
(:Fund)-[h:HAS_SECTOR_ALLOCATION {weight, date}]->(:Sector {name})
(:Fund)-[h:HAS_REGION_ALLOCATION {weight, date}]->(:Region {name})

# Holdings Structure
(:Fund)-[:HAS_PORTFOLIO]->(:Portfolio {date, seriesId})
(:Portfolio)-[:HAS_HOLDING {shares, marketValue, weight, fairValueLevel, isRestricted, payoffProfile}]->(:Holding {
    name, ticker, isin, lei, category, category_desc, issuer_category, businessAddress})
(:Portfolio)-[:EXTRACTED_FROM]->(:Document)
# Financial Highlights
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {
    turnover,                    # Portfolio turnover rate (percentage)
    expenseRatio,                # Total expense ratio (percentage)
    totalReturn,                 # Total return for the period (percentage)
    netAssets,                   # Total net assets under management 
    netAssetsValueBeginning,     # Price of one share at period start
    netAssetsValueEnd,           # Price of one share at period end
    netIncomeRatio               # Net investment income ratio (percentage)

# ============================================================
# === COMPANY STRUCTURE (10-K Filings) ===
# ============================================================

# === COMPANY NODE PROPERTIES ===
(:Company {
    ticker,              # Stock ticker symbol like 'AAPL', 'MSFT'
    name,                # Company name like 'Apple Inc.'
    cik                  # SEC Central Index Key
})
# === COMPANY RELATIONSHIPS ===
# Funds can hold Company stocks
(:Holding {ticker})-[:REPRESENTS]->(:Company {ticker})

(:Company)-[:HAS_FILING {date}]->(:Filing10K)
(:Filing10K)-[:EXTRACTED_FROM]->(:Document)

# 10-K Section Types (all inherit from :Section base label)
(:Filing10K)-[:HAS_RISK_FACTOR_CHUNK]->(:Section:RiskFactor {id, text, embedding})
(:Filing10K)-[:HAS_BUSINESS_INFORMATION_CHUNK]->(:Section:BusinessInformation {id, text, embedding})
(:Filing10K)-[:HAS_LEGAL_PROCEEDING_CHUNK]->(:Section:LegalProceeding {id, text, embedding})
(:Filing10K)-[:HAS_MANAGEMENT_DISCUSSION_CHUNK]->(:Section:ManagemetDiscussion {id, text, embedding})
# NOTE: Properties node uses 'fullText', not 'text'
(:Filing10K)-[:HAS_PROPERTIES_CHUNK]->(:Section:Properties {id, fullText, embedding})

(:Filing10K)-[:HAS_FINACIALS]->(:Section:Financials {incomeStatement, balanceSheet, cashFlow, fiscalYear})

# === COMPANY PEOPLE & COMPENSATION ===
(:Company)-[:HAS_CEO {ceoCompensation, ceoActuallyPaid, date}]->(:Person {name})
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage {totalCompensation, shareholderReturn, date})
(:CompensationPackage)-[:AWARDED_BY]->(:Company)
(:CompensationPackage)-[:DISCLOSED_IN]->(:Document)
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction {
    position, transactionType, shares, price, value, remainingShares
})
(:InsiderTransaction)-[:MADE_BY]->(:Person {name})
(:InsiderTransaction)-[:EXTRACTED_FROM]->(:Document)

# === FINANCIAL METRICS & SEGMENTS ===
(:Section:Financials)-[:HAS_METRIC]->(:FinancialMetric {label, value})
(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment {label, value, percentage})

# === IMPORTANT NOTES ===
# 1. netAssets is DIRECTLY on Fund node, not in a separate FinancialHighlight node
# 2. numberHoldings property already contains the count but you can recalculate
# 3. turnoverRate is absolute (2 = 2%, not 0.02)
# 4. Use 'ticker' for symbols (VTI), 'name' for full names
# 5. Vector indexes exist on embedding properties for semantic search
# 6. Fulltext indexes exist on name properties for fuzzy/partial name matchings USE THEM before the MATCH
# 7. NEVER generate incomplete property filters like (n:Label {name}). Only use property filters with values like (n:Label {name: 'Value'}).
# 8. If you do not know the value of a property, DO NOT include it in the curly braces {}.
# 9. Whenever it exists, return the document information from which it has been extracted.
# 10. Do not forget the r in the relationship name like (n:Label)-[r:RELATIONSHIP]->(m:Label) and use it in the return statement.
"""