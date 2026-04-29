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
    cik,                 # Central Index Key (SEC identifier)
    securityExchange     # Exchange like 'NASDAQ', 'NYSE'
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
# Document node — MERGE key is accession_number
(:Fund)-[:EXTRACTED_FROM]->(:Document {accession_number, url, form, filing_date, reporting_date})
# Profile (versioned by year)
(:Fund)-[:DEFINED_BY {year}]->(:Profile {summaryProspectus})
# Profile sections use multi-labeled Section nodes (filter by label, not property)
(:Profile)-[:HAS_SECTION]->(:Section:Objective {text, title, embedding})
(:Profile)-[:HAS_SECTION]->(:Section:PerformanceCommentary {text, title, embedding})
(:Profile)-[:HAS_SECTION]->(:Section:RiskFactor {text, title})
(:Profile)-[:HAS_SECTION]->(:Section:Strategy {text, title})
# Risk/Strategy sections have child chunks:
(:Section)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Profile)-[:EXTRACTED_FROM]->(:Document)
# Charts/Images
(:Fund)-[:HAS_CHART {year}]->(:Image {category, svg, title})
# Tables
(:Fund)-[:HAS_TABLE {year}]->(:Table {content, title})

# Management Team (year on the relationship)
(:Fund)-[:MANAGED_BY {year}]->(:Person {name})

# Average Returns (versioned by year)
(:Fund)-[:HAS_AVERAGE_RETURNS {year}]->(:AverageReturns {return1y, return5y, return10y, returnInception})

# Allocations (by year)
(:Fund)-[h:HAS_SECTOR_ALLOCATION {weight, year}]->(:Sector {name})
(:Fund)-[h:HAS_REGION_ALLOCATION {weight, year}]->(:Region {name})

# Holdings Structure
(:Fund)-[:HAS_PORTFOLIO]->(:Portfolio {date, seriesId})
(:Portfolio)-[:HAS_HOLDING {shares, marketValue, weight, fairValueLevel, isRestricted, payoffProfile}]->(:Holding {
    name, ticker, isin, lei, category, category_desc, issuer_category, businessAddress})
(:Portfolio)-[:EXTRACTED_FROM]->(:Document)
# Financial Highlights — IMPORTANT: 'year' is on the RELATIONSHIP, not the node!
# ALWAYS use: (:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) and access r.year
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {
    year                         # ⚠️ CRITICAL: Year is a RELATIONSHIP property! Use r.year, NOT fh.year
}]->(:FinancialHighlight {
    turnover,                    # Portfolio turnover rate (percentage)
    expenseRatio,                # Total expense ratio (percentage)
    totalReturn,                 # Total return for the period (percentage)
    netAssets,                   # Total net assets under management 
    netAssetsValueBeginning,     # Price of one share at period start
    netAssetsValueEnd,           # Price of one share at period end
    netIncomeRatio,              # Net investment income ratio (percentage)
    numberHoldings,              # Total number of holdings (integer) - USE THIS, not count(h)!
    advisoryFees,                # Advisory fees (numeric)
    costsPer10k                  # Costs per $10,000 invested (numeric)
})
# Example: MATCH (f:Fund {ticker: 'VTI'})-[r:HAS_FINANCIAL_HIGHLIGHT]->(fh:FinancialHighlight) RETURN r.year, fh.totalReturn

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

(:Company)-[:REPORTS_IN {year}]->(:Filing10K)
(:Filing10K)-[:EXTRACTED_FROM]->(:Document)

# 10-K Section Types (TWO-LEVEL ARCHITECTURE)
# Level 1: Section nodes store FULL TEXT — unified HAS_SECTION relationship
# Level 2: Chunk:SectionChunk nodes store CHUNKED TEXT for fine-grained retrieval
# Each section has: id, title, text (full), sectionType, secItem
(:Filing10K)-[:HAS_SECTION]->(:Section {id, title, text, sectionType, secItem})
# Section additional labels: RiskFactor, BusinessInformation, LegalProceeding, ManagementDiscussion, Properties
# sectionType values: 'risk_factors', 'business_info', 'legal_proceedings', 'mda', 'properties'
# secItem values: 'Item 1A', 'Item 1', 'Item 3', 'Item 7', 'Item 2'

# Section chunks for fine-grained retrieval (linked to parent Section)
(:Section)-[:HAS_CHUNK]->(:Chunk:SectionChunk {title, text, embedding, chunkType, chunkIndex, subsection})

(:Filing10K)-[:HAS_FINANCIALS]->(:Section:Financials {incomeStatement, balanceSheet, cashFlow, fiscalYear})

# === COMPANY PEOPLE & COMPENSATION ===
(:Company)-[:HAS_CEO {ceoCompensation, ceoActuallyPaid, date}]->(:Person {name})
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage {totalCompensation, shareholderReturn, date})
(:CompensationPackage)-[:AWARDED_BY]->(:Company)
(:CompensationPackage)-[:DISCLOSED_IN]->(:Document)
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction {
    transactionDate, position, transactionType, shares, price, value, remainingShares
})
(:InsiderTransaction)-[:MADE_BY]->(:Person {name})
(:InsiderTransaction)-[:EXTRACTED_FROM]->(:Document)

# === FINANCIAL METRICS & SEGMENTS ===
(:Section:Financials)-[:HAS_METRIC]->(:FinancialMetric {label, value})
(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment {label, value, percentage})

# === IMPORTANT NOTES ===
# 1. netAssets, numberHoldings, expenseRatio, advisoryFees, costsPer10k are on FinancialHighlight, NOT on Fund
# 2. numberHoldings property already contains the count but you can recalculate
# 3. turnoverRate is absolute (2 = 2%, not 0.02)
# 4. Use 'ticker' for symbols (VTI), 'name' for full names
# 5. Vector indexes exist on embedding properties for semantic search
# 6. Fulltext indexes exist on name properties for fuzzy/partial name matchings USE THEM before the MATCH
# 7. NEVER generate incomplete property filters like (n:Label {name}). Only use property filters with values like (n:Label {name: 'Value'}).
# 8. If you do not know the value of a property, DO NOT include it in the curly braces {}.
# 9. Whenever it exists, return the document information from which it has been extracted.
# 10. Do not forget the r in the relationship name like (n:Label)-[r:RELATIONSHIP]->(m:Label) and use it in the return statement.
# 11. ⚠️ CRITICAL: For HAS_FINANCIAL_HIGHLIGHT, 'year' is on the RELATIONSHIP (r.year), NOT on the FinancialHighlight node (fh.year does NOT exist)!
"""