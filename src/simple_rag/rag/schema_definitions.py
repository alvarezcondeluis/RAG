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
# === FUND NODE PROPERTIES ===
(:Fund {
    ticker,              # Symbol like 'VTI' (use for matching symbols)
    name,                
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
# Document node — MERGE key is accessionNumber
(:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, form, filingDate, reportingDate})
# Profile (versioned by year)
(:Fund)-[:DEFINED_BY {year}]->(:Profile {summaryProspectus}) # Contains the whole text content of the prospectus
# Profile sections use multi-labeled Section nodes (filter by label, not property)
# IMPORTANT — embeddings on Profile sections:
#   - :Section:Objective       → embedding ON THE SECTION (no chunks). Use profileObjectiveIndex.
#   - :Section:Strategy        → no embedding on section; child :Chunk has embedding. Use profileChunkIndex.
#   - :Section:RiskFactor      → no embedding on section; child :Chunk has embedding. Use profileChunkIndex.
#   - :Section:PerformanceCommentary → no embedding anywhere; only structural retrieval.
(:Profile)-[:HAS_SECTION]->(:Section:Objective {text, title, embedding})
(:Profile)-[:HAS_SECTION]->(:Section:PerformanceCommentary {text, title})
(:Profile)-[:HAS_SECTION]->(:Section:RiskFactor {text, title})
(:Profile)-[:HAS_SECTION]->(:Section:Strategy {text, title})
# Chunks under Profile Strategy / RiskFactor sections (these are the embedded units)
(:Section:Strategy)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Section:RiskFactor)-[:HAS_CHUNK]->(:Chunk {text, embedding})
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
(:Fund)-[:HAS_PORTFOLIO]->(:Portfolio {date, seriesId, count})  # count = number of holdings
(:Portfolio)-[:HAS_HOLDING {shares, marketValue, weight, fairValueLevel, isRestricted, payoffProfile}]->(:Holding {
    name, ticker, isin, lei, country, category, categoryDesc, issuerCategory, businessAddress})
(:Holding)-[:OF_ASSET_TYPE]->(:AssetCategory {code, name, category, subcategory})
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
    advisoryFees                 # Advisory fees (numeric)
})
# ⚠️ numberHoldings is NOT on FinancialHighlight. Use Portfolio.count:
# MATCH (f:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.count AS numberHoldings

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

# 10-K Section Types — sections store {text, title}; embeddings live on the child :Chunk nodes.
# Note: 'ManagemetDiscussion' is intentionally misspelled (matches actual label in DB).
(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:ManagemetDiscussion {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {text, title})

# Section chunks for fine-grained retrieval — these are the embedded units.
# All five Section types above expose: (:Section)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:RiskFactor)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:BusinessInformation)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:LegalProceeding)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:ManagemetDiscussion)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:Properties)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
# === FINANCIAL METRICS & SEGMENTS ===
(:Filing10K)-[:HAS_FINANCIALS]->(:Section:Financials {incomeStatement, balanceSheet, cashFlow, fiscalYear})
(:Section:Financials)-[:HAS_METRIC]->(:FinancialMetric {label, value})
(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment {label, value, percentage})

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


# ============================================================
# === VECTOR INDEXES (semantic search via $queryVector parameter) ===
# ============================================================
# Use these when the question asks about MEANING / TOPIC / CONCEPT of narrative
# text. Pattern: CALL db.index.vector.queryNodes('<indexName>', $k, $queryVector)
#
#   filing10kChunkIndex
#     → :Chunk under any 10-K section (RiskFactor, BusinessInformation,
#       LegalProceeding, ManagemetDiscussion, Properties).
#     Returns Chunk nodes — traverse <-[:HAS_CHUNK]-(:Section) to get the
#     parent Section, and <-[:HAS_SECTION]-(:Filing10K)<-[:REPORTS_IN]-(:Company)
#     to get the company. Filter by Section label (e.g. :RiskFactor) inside MATCH.
#
#   profileChunkIndex
#     → :Chunk under :Section:Strategy or :Section:RiskFactor (Profile side).
#     Traverse <-[:HAS_CHUNK]-(:Section) <-[:HAS_SECTION]-(:Profile) <-[:DEFINED_BY]-(:Fund).
#
#   profileObjectiveIndex
#     → :Section:Objective directly (embedding lives on the section, no chunks).
#     Traverse <-[:HAS_SECTION]-(:Profile) <-[:DEFINED_BY]-(:Fund).
#
# DECISION RULE:
#   - Numeric/property/relationship questions → use plain MATCH patterns.
#   - Narrative-content questions ("what risks…", "describe the strategy…") →
#     use vector index call. Hybrid is allowed: filter by structure (e.g. by
#     ticker), then vector-search inside the filtered subgraph.
#   - :Section:PerformanceCommentary has NO embeddings — use only text search
#     or structural retrieval, not vector search.

# === IMPORTANT NOTES ===
# 1. netAssets, expenseRatio, advisoryFees are on FinancialHighlight, NOT on Fund
# 2. numberHoldings is on Portfolio.count — use MATCH (f)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.count
# 3. turnover is absolute (2 = 2%, not 0.02)
# 4. Use 'ticker' for symbols (VTI), 'name' for full names
# 5. Vector indexes exist on embedding properties for semantic search
# 6. Fulltext indexes exist on name properties for fuzzy/partial name matchings USE THEM before the MATCH
# 7. NEVER generate incomplete property filters like (n:Label {name}). Only use property filters with values like (n:Label {name: 'Value'}).
# 8. If you do not know the value of a property, DO NOT include it in the curly braces {}.
# 9. ⚠️ CRITICAL: Whenever it exists, return the document information from which it has been extracted.
# 10. Do not forget the r in the relationship name like (n:Label)-[r:RELATIONSHIP]->(m:Label) and use it in the return statement.
"""