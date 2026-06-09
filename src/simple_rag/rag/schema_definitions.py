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
    ticker,              # Symbol like 'AAPL' (use for matching symbols)
    name,                
    cik,                 # Central Index Key (SEC identifier)
    securityExchange     # Exchange like 'NASDAQ', 'NYSE'
})
# FULL-TEXT INDEXES (use for fuzzy/partial name matching):
# - Provider.name -> use CALL db.index.fulltext.queryNodes('providerNameIndex', 'search_term')
# - Trust.name -> use CALL db.index.fulltext.queryNodes('trustNameIndex', 'search_term')
# - Fund.name -> use CALL db.index.fulltext.queryNodes('fundNameIndex', 'search_term')
# - Person.name -> use CALL db.index.fulltext.queryNodes('personNameIndex', 'search_term')
# For exact ticker matching, use {ticker: 'TICKER'} (no index needed)

# === FUND RELATIONSHIPS ===
# Share Classes
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
# Document node — MERGE key is accessionNumber
(:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})
# Profile (versioned by year)
(:Fund)-[:DEFINED_BY {year}]->(:Profile {summaryProspectus}) # Contains the whole text content of the prospectus
# Profile sections use multi-labeled Section nodes (filter by label, not property)
# IMPORTANT — embeddings on Profile sections:
#   - :Section:Objective       → embedding ON THE SECTION (no chunks). Use profileObjectiveIndex.
#   - :Section:Strategy        → no embedding on section; child :Chunk has embedding. Use chunkEmbeddingIndex.
#   - :Section:Risk             → no embedding on section; child :Chunk has embedding. Use chunkEmbeddingIndex.
#   - :Section:PerformanceCommentary → no embedding anywhere; only structural retrieval.
(:Profile)-[:HAS_SECTION]->(:Section:Objective {text, title, embedding})
(:Profile)-[:HAS_SECTION]->(:Section:PerformanceCommentary {text, title})
# ⚠️ WARNING: Risk and Strategy sections have text: NULL on the Section node.
# Full content lives ONLY in child :Chunk nodes. NEVER return s.text for these — always use HAS_CHUNK.
# CORRECT PATTERN (Fund Profile Risk):
#   MATCH (f:Fund {ticker: $ticker})-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:Risk)-[:HAS_CHUNK]->(chunk:Chunk)
#   RETURN s.title AS sectionTitle, chunk.text AS chunkContent ORDER BY chunk.id ASC
# CORRECT PATTERN (Fund Profile Strategy):
#   MATCH (f:Fund {ticker: $ticker})-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:Strategy)-[:HAS_CHUNK]->(chunk:Chunk)
#   RETURN s.title AS sectionTitle, chunk.text AS chunkContent ORDER BY chunk.id ASC
(:Profile)-[:HAS_SECTION]->(:Section:Risk {text: null, title})       # text is NULL — use HAS_CHUNK
(:Profile)-[:HAS_SECTION]->(:Section:Strategy {text: null, title})   # text is NULL — use HAS_CHUNK
# Chunks under Profile Strategy / Risk sections (these are the embedded units)
(:Section:Strategy)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Section:Risk)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Profile)-[:EXTRACTED_FROM]->(:Document)
# Charts/Images
(:Fund)-[:HAS_CHART {year}]->(:Image {category, svg, title})
# Tables
(:Fund)-[:HAS_TABLE {year}]->(:Table {content, title})

# Management Team (year on the relationship)
(:Fund)-[:MANAGED_BY {year}]->(:Person {name})

# Average Returns (versioned by year)
(:Fund)-[:HAS_AVERAGE_RETURNS {year}]->(:AverageReturns {return1y, return5y, return10y, returnInception})
# Filter by the ones that are not null when asked for specific return
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

# === COMPANY STRUCTURE (10-K Filings) ===

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
# Year property is in the relationship REPORTS_IN not in the Filing10K
(:Filing10K)-[:EXTRACTED_FROM]->(:Document)

# ⚠️ WARNING — 10-K Section text storage rules:
#   Properties  → text IS populated on the Section node. Use s.text directly.
#   RiskFactor, BusinessInformation, LegalProceeding, ManagementDiscussion
#               → text is NULL on the Section node. Full content is in child :Chunk nodes.
#                 ALWAYS traverse HAS_CHUNK and return chunk.text for these sections.
#
# CORRECT PATTERN for RiskFactor / BusinessInformation / LegalProceeding / ManagementDiscussion:
#   MATCH (c:Company {ticker: 'AAPL'})-[r:REPORTS_IN]->(f:Filing10K)
#         -[:HAS_SECTION]->(s:Section:RiskFactor)-[:HAS_CHUNK]->(chunk:Chunk)
#   RETURN s.title AS sectionTitle, chunk.text AS chunkContent, r.year AS year
#   ORDER BY r.year DESC, chunk.id ASC
#
# CORRECT PATTERN for Properties (text IS on the Section):
#   MATCH (c:Company {ticker: 'MSFT'})-[:REPORTS_IN]->(f:Filing10K)-[:HAS_SECTION]->(s:Section:Properties)
#   RETURN s.title AS sectionTitle, s.text AS properties
(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {text: null, title})          # text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {text: null, title}) # text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {text: null, title})     # text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:ManagementDiscussion {text: null, title})# text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {text, title})                # text IS populated

# Section chunks for fine-grained retrieval — these are the embedded units.
# All five Section types above expose: (:Section)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:RiskFactor)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:BusinessInformation)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:LegalProceeding)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:ManagementDiscussion)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
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
#   chunkEmbeddingIndex
#     → :Chunk under any 10-K section (RiskFactor, BusinessInformation,
#       LegalProceeding, ManagementDiscussion, Properties).
#     Returns Chunk nodes — traverse <-[:HAS_CHUNK]-(:Section) to get the
#     parent Section, and <-[:HAS_SECTION]-(:Filing10K)<-[:REPORTS_IN]-(:Company)
#     to get the company. Filter by Section label (e.g. :RiskFactor) inside MATCH.
#
#   chunkEmbeddingIndex (also used for Profile chunks — same physical index)
#     → :Chunk under :Section:Strategy or :Section:RiskFactor (Profile side).
#     Traverse <-[:HAS_CHUNK]-(:Section) <-[:HAS_SECTION]-(:Profile) <-[:DEFINED_BY]-(:Fund).
#
#   profileObjectiveIndex
#     → :Section:Objective directly (embedding lives on the section, no chunks).
#     Traverse <-[:HAS_SECTION]-(:Profile) <-[:DEFINED_BY]-(:Fund).

# ENTITY FILTERS (critical — applies to ALL query types, not just vector search):
#   Add a ticker or name filter ONLY IF the ticker/name appears in the Entity Context block above.
#   If Entity Context is empty → NEVER add any ticker or name filter. Search globally.
#   NEVER infer a ticker from the schema examples or from your training data.
#   BAD: `WHERE f.name CONTAINS 'Total Market'` (name fragment, not from Entity Context)
#   BAD: `(f:Fund {ticker: 'EFAV'})` when Entity Context is empty or doesn't mention EFAV
#   GOOD: `(f:Fund {ticker: 'XYZ'})` only when Entity Context explicitly contains ticker XYZ

# === IMPORTANT NOTES ===
# 1. netAssets, expenseRatio, advisoryFees are on FinancialHighlight, NOT on Fund
# 2. numberHoldings is on Portfolio.count — use MATCH (f)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.count
# 3. turnover is absolute (2 = 2%, not 0.02)
# 4. NEVER generate incomplete property filters like (n:Label {name}). Only use property filters with values like (n:Label {name: 'Value'}).
# 5. If you do not know the value of a property, DO NOT include it in the curly braces {}.
# 6 RELATIONSHIP VARIABLES: Only assign a variable name to a relationship edge (e.g., -[r:RELATIONSHIP]->) if you explicitly need to filter or return a property stored directly on that edge (such as 'year' or 'weight'). Otherwise, keep relationships anonymous (-[:RELATIONSHIP]->) to prevent syntax errors.
"""


DETAILED_SCHEMA_V2 = """
# ═══════════════════════════════════════════════════════════════
# FUND MANAGEMENT
# ═══════════════════════════════════════════════════════════════
(:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund {
    ticker,           # exact symbol — use {ticker: 'TICKER'} directly, no index needed
    name,             # full name   — use fundNameIndex for fuzzy/partial search
    cik,
    securityExchange  # 'NASDAQ', 'NYSE', …
})
# Full-text indexes: fundNameIndex | providerNameIndex | trustNameIndex | personNameIndex
# Usage: CALL db.index.fulltext.queryNodes('fundNameIndex', 'search term') YIELD node AS f, score

(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
(:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})
(:Fund)-[:DEFINED_BY {year}]->(:Profile {summaryProspectus})
(:Fund)-[:HAS_CHART {year}]->(:Image {category, svg, title})
(:Fund)-[:HAS_TABLE {year}]->(:Table {content, title})
(:Fund)-[:MANAGED_BY {year}]->(:Person {name})
(:Fund)-[r:HAS_AVERAGE_RETURNS {year}]->(:AverageReturns {return1y, return5y, return10y, returnInception})
# ⚠️ returnInception / return1y / return5y / return10y can be NULL — ALWAYS add WHERE ar.returnXxx IS NOT NULL when filtering or ordering by these fields
(:Fund)-[h:HAS_SECTOR_ALLOCATION {weight, year}]->(:Sector {name})
(:Fund)-[h:HAS_REGION_ALLOCATION {weight, year}]->(:Region {name})
(:Fund)-[:HAS_PORTFOLIO]->(:Portfolio {date, seriesId, count})  # count = total number of holdings

# ⚠️ year lives on the RELATIONSHIP — use r.year, NOT fh.year
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {
    expenseRatio,            # ⚠️ 0.0 = missing — ALWAYS add AND fh.expenseRatio > 0
    turnover,                # absolute % — value 2 means 2%, not 0.02
    totalReturn,
    netAssets,
    netAssetsValueBeginning, # NAV per share at period start
    netAssetsValueEnd,       # NAV per share at period end
    netIncomeRatio,
    advisoryFees
})
# ⚠️ numberHoldings is NOT on FinancialHighlight — use (:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.count

# PROFILE SECTIONS
(:Profile)-[:HAS_SECTION]->(:Section:Objective {text, title, embedding})         # text ✓ | use profileObjectiveIndex
(:Profile)-[:HAS_SECTION]->(:Section:PerformanceCommentary {text, title})         # text ✓ | no chunks
(:Profile)-[:HAS_SECTION]->(:Section:Risk {title})                          # text NULL → must use HAS_CHUNK
(:Profile)-[:HAS_SECTION]->(:Section:Strategy {title})                            # text NULL → must use HAS_CHUNK
(:Section:Risk)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Section:Strategy)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Profile)-[:EXTRACTED_FROM]->(:Document)

# PORTFOLIO & HOLDINGS
(:Portfolio)-[:EXTRACTED_FROM]->(:Document)
(:Portfolio)-[:HAS_HOLDING {shares, marketValue, weight, fairValueLevel, isRestricted, payoffProfile}]->
    (:Holding {name, ticker, isin, lei, country, category, categoryDesc, issuerCategory, businessAddress})
(:Holding)-[:OF_ASSET_TYPE]->(:AssetCategory {code, name, category, subcategory})
(:Holding {ticker})-[:REPRESENTS]->(:Company {ticker})  # link fund holdings to Company nodes

# ═══════════════════════════════════════════════════════════════
# COMPANY (10-K FILINGS)
# ═══════════════════════════════════════════════════════════════
(:Company {ticker, name, cik})-[r:REPORTS_IN {year}]->(:Filing10K)-[:EXTRACTED_FROM]->(:Document)
# companyNameIndex: CALL db.index.fulltext.queryNodes('companyNameIndex', 'Apple') YIELD node AS c, score
#Year property is in the relationship REPORTS_IN not in the Filing10K
# 10-K SECTIONS — text storage per section type:
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {text, title})             # text ✓  → return s.text directly
(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {title})                   # text NULL → traverse HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {title})          # text NULL → traverse HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {title})              # text NULL → traverse HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:ManagementDiscussion {title})         # text NULL → traverse HAS_CHUNK
(:Section)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})

(:Filing10K)-[:HAS_FINANCIALS]->(:Section:Financials {incomeStatement, balanceSheet, cashFlow, fiscalYear})
(:Section:Financials)-[:HAS_METRIC]->(:FinancialMetric {label, value})-[:HAS_SEGMENT]->(:Segment {label, value, percentage})

# COMPANY PEOPLE & COMPENSATION
(:Company)-[r:HAS_CEO {ceoCompensation, ceoActuallyPaid, date}]->(:Person {name})
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage {totalCompensation, shareholderReturn, date})
(:CompensationPackage)-[:AWARDED_BY]->(:Company)
(:CompensationPackage)-[:DISCLOSED_IN]->(:Document)
# ⚠️ HAS_INSIDER_TRANSACTION: properties are on the InsiderTransaction NODE, not the relationship
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction {
    transactionDate, position, transactionType, shares, price, value, remainingShares
})-[:MADE_BY]->(:Person {name})
(:InsiderTransaction)-[:EXTRACTED_FROM]->(:Document)

# ═══════════════════════════════════════════════════════════════
# VECTOR INDEXES  (use when question asks about MEANING / TOPIC)
# ═══════════════════════════════════════════════════════════════
# chunkEmbeddingIndex   → :Chunk nodes under any 10-K section OR Profile Strategy/Risk
# profileObjectiveIndex → :Section:Objective directly (embedding on the section node, no chunks)
#
# Example (10-K chunk search):
#   CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 10, $queryVector) YIELD node AS chunk, score
#   MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:RiskFactor)<-[:HAS_SECTION]-(f:Filing10K)
#         <-[r:REPORTS_IN]-(c:Company {ticker: 'AAPL'})
#   RETURN chunk.text AS text, r.year AS year, score ORDER BY score DESC LIMIT 5
#
# Entity filter: add ONLY IF the ticker appears in the Entity Context block. If Entity Context is empty → no filter.

# ═══════════════════════════════════════════════════════════════
# CRITICAL RULES
# ═══════════════════════════════════════════════════════════════
# R1  expenseRatio     — always add AND fh.expenseRatio > 0  (0.0 means missing data, not free)
# R2  year on FH       — always [r:HAS_FINANCIAL_HIGHLIGHT {year}] → r.year, never fh.year
# R3  numberHoldings   — Portfolio.count, not any FinancialHighlight property
# R4  turnover         — absolute value: 2 = 2%, not 0.02
# R5  rel variable     — only use -[r:REL]-> if you need r.property in RETURN/WHERE/ORDER BY
# R6  property filters — never (n:Label {prop}); always (n:Label {prop: 'Value'})
# R7  NULL in ORDER BY — add WHERE prop IS NOT NULL before sorting numeric fields
# R8  Properties text  — s.text is populated; all other 10-K sections need HAS_CHUNK
"""