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

# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL CYPHER RULES (apply to ALL schema slices)
# ═══════════════════════════════════════════════════════════════════════════
UNIVERSAL_RULES = """
UNIVERSAL CYPHER RULES (apply to ALL queries):

1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, markdown, or ```cypher blocks.

2. UNIQUE VARIABLES: NEVER use the same variable name for both a relationship and a node. To prevent this, ALWAYS leave relationships anonymous (e.g., -[:HAS_PORTFOLIO]->(p:Portfolio)) unless you specifically need to extract a property stored on the edge itself.
3. SUBQUERY SYNTAX: Use WHERE EXISTS { MATCH ... } or WITH, NOT SQL-style IN (MATCH...).

4. AGGREGATIONS & GROUPING: For global calculations (highest, average, total), DO NOT include
   row-specific identifiers (ticker, name, year) in RETURN — triggers unwanted GROUP BY.

5. EXTRA PROPERTIES: Always return identifying properties (ticker, name) and relationship properties.

6. SCHEMA STRICTNESS: Use property names EXACTLY as provided. No hallucinations or shortcuts.

7. FILTERING: WHERE clauses follow MATCH or WITH, NEVER after RETURN. Use CONTAINS for text, >, < for numeric.

8. DUPLICATES: Use RETURN DISTINCT if a MATCH traverses non-returned nodes.

9. CONTEXT MATCHING: Use entity names from Entity Context EXACTLY as written.

10. FEW-SHOT REPLICATION: If example marked [★ VERY SIMILAR], replicate its structure exactly.

11. FULLTEXT INDEXES: If entity score ≈ 100, use direct lookup {ticker: 'X'}, NOT the index.

12. LATEST: Use ORDER BY date DESC LIMIT 1, NOT complex NOT EXISTS subqueries.

13. TICKER: When query contains a ticker, ALWAYS use it exactly as stated.

14. NO YEAR FILTER UNLESS ASKED: Do NOT add year/date filters unless question explicitly asks.

15. DIVISION BY ZERO: Use CASE WHEN denominator = 0 THEN 0 ELSE (numerator * 100.0 / denominator) END.

16. INCOMPLETE FILTERS: NEVER generate {name} without value. Only {name: 'value'}.

17. CASE-SENSITIVE MATCHING: Neo4j CONTAINS is case-sensitive. When searching by user-provided
   fund/company names (not tickers), use toLower(): WHERE toLower(f.name) CONTAINS 'bond'.
   Only skip toLower() when matching exact known values from Entity Context.

18. SUPERLATIVE/AGGREGATE QUERIES: For "which fund has the highest/lowest/best/worst..."
   queries, search ALL entities globally. Do NOT filter by a specific ticker or entity name
   unless the question explicitly names one. Ignore Entity Context for these queries.
"""

SCHEMA_SLICES_V1 = {
    "not_related": None,  # No schema needed for out-of-scope queries

    "fund_basic": """
# FUND BASIC PERFORMANCE & METRICS
# ⚠️  See UNIVERSAL_RULES at top of file for 16 core Cypher rules

(:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund {
    ticker,              # Symbol like 'VTI'
    name,                # Full fund name
    cik,                 # SEC Central Index Key
    securityExchange     # Exchange like 'NASDAQ', 'NYSE'
})
# ⚠️ netAssets, expenseRatio, advisoryFees are on FinancialHighlight, NOT on Fund
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
(:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})

# Financial Highlights — year is on the RELATIONSHIP, not the node
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {
    turnover,                # Portfolio turnover (%) — absolute: 2 = 2%, not 0.02
    expenseRatio,            # Expense ratio (%)
    totalReturn,             # Total return for the period (%)
    netAssets,               # Total net assets under management
    netAssetsValueBeginning, # Price of one share at period start
    netAssetsValueEnd,       # Price of one share at period end
    netIncomeRatio,          # Net investment income ratio (%)
    advisoryFees             # Advisory fees (numeric)
})

# Average Returns (multi-year performance)
(:Fund)-[r:HAS_AVERAGE_RETURNS {year}]->(:AverageReturns {
    return1y,            # 1-year annualized return
    return5y,            # 5-year annualized return
    return10y,           # 10-year annualized return
    returnInception      # Since-inception annualized return
})

# Charts (year on the relationship)
(:Fund)-[:HAS_CHART {year}]->(:Image {
    category,            # Chart category (e.g. 'performance')
    svg,                 # SVG content of the chart
    title                # Chart title
})

# Tables (year on the relationship)
(:Fund)-[:HAS_TABLE {year}]->(:Table {
    content,             # Table content (text/JSON)
    title                # Table title
})

FUND_BASIC SPECIFIC RULES:
- Fulltext indexes: fundNameIndex, providerNameIndex, trustNameIndex (YIELD node AS var, use exactly)
- numberHoldings → use Portfolio.count, NOT COUNT(h)
- year is on relationship: use r.year in HAS_FINANCIAL_HIGHLIGHT
- turnover is absolute: 2 = 2%, not 0.02
- Always include relationship variable and use in RETURN
- ALWAYS return source document via EXTRACTED_FROM
- For fund/provider/trust name searches, use fulltext BEFORE MATCH

FUND_BASIC ADDITIONAL RULES (beyond UNIVERSAL_RULES):
- STRICT SCHEMA ALIGNMENT: `netAssets`, `expenseRatio`, `advisoryFees` are on FinancialHighlight, NOT Fund
- COMPARING: Use WHERE f.ticker IN ['VTI', 'VOO'] for multi-fund comparisons
- HISTORICAL GROWTH: MATCH all highlights, ORDER BY year, COLLECT, then compare indices
- AVERAGE RETURNS: Use return1y, return5y, return10y properties (not calculated)
""",

    "fund_portfolio": """
# FUND PORTFOLIO COMPOSITION & ALLOCATIONS
# ⚠️  See UNIVERSAL_RULES at top of file for 16 core Cypher rules

(:Fund {
    ticker,              # Symbol like 'VTI'
    name,                # Full fund name
    securityExchange     # Exchange like 'NASDAQ', 'NYSE'
})
# ⚠️ numberHoldings is NOT on Fund. Use Portfolio.count:
# MATCH (f:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.count AS numberHoldings

# Sector allocations (Direct relationship from Fund, NOT Portfolio!)
(:Fund)-[sa:HAS_SECTOR_ALLOCATION {weight, year}]->(:Sector {
    name                 # Sector name (Technology, Healthcare, etc.)
})

# Regional/geographic allocations (Direct relationship from Fund, NOT Portfolio!)
(:Fund)-[ra:HAS_REGION_ALLOCATION {weight, year}]->(:Region {
    name                 # Region name (North America, Asia-Pacific, etc.)
})

# Tables — summaries of allocations, holdings, compositions
# Common table titles:
#   - "Sector Allocation" (sector weights)
#   - "Top Holdings" (largest positions by weight)
#   - "Geographic Allocation" (regional/country breakdown)
#   - "Portfolio Composition" (asset type breakdown)
#   - "Maturity Allocation" (for bond funds)
#   - "Credit Rating Distribution" (credit quality breakdown)
#   - "Industry Allocation" (industry-level sector detail)
#   - "Top Issuer Allocation" (largest issuer concentrations)

(:Fund)-[rt:HAS_TABLE {year}]->(:Table {
    title,               # Table title (see list above)
    content              # Table content (text or JSON)
})
# Portfolio structure snapshot
(:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio {
    date,                # Portfolio snapshot date
    seriesId,            # Series identifier
    count                # Number of holdings — use this, do NOT use COUNT(h)
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
    country,             # Country of issuer
    category,            # Asset category code
    categoryDesc,        # Category description
    issuerCategory,      # Issuer category
    businessAddress      # Business address
})
# Asset type classification
(h)-[:OF_ASSET_TYPE]->(:AssetCategory {
    code,                # Asset category code
    name,                # Asset category name
    category,            # Broad asset type
    subcategory          # Specific subtype
})

# Link holdings to companies
(h)-[:REPRESENTS]->(:Company {ticker, name})

(p)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})

FUND_PORTFOLIO SPECIFIC RULES:
- SECTOR & REGION DIRECT MATCHING: Sector and Region nodes connect DIRECTLY to the (:Fund) node. They do NOT pass through or hop through a (:Portfolio) node.
  * ⚠️ BAD:  MATCH (f:Fund)-[:HAS_PORTFOLIO]->(p)-[:HAS_SECTOR_ALLOCATION]->(s:Sector)
  * ✅ GOOD: MATCH (f:Fund)-[sa:HAS_SECTOR_ALLOCATION]->(s:Sector)
- Use p.count for holdings, NOT COUNT(h)
- weight, marketValue, payoffProfile are on relationship [r:HAS_HOLDING], NOT Holding node
- year is on HAS_SECTOR_ALLOCATION / HAS_REGION_ALLOCATION relationships
- Fulltext index: holdingNameIndex (YIELD node AS h, score)
- ALWAYS return source document via EXTRACTED_FROM when matching historical Portfolio snapshots
- INLINE MATH: NEVER use math operators in curly braces. Bad: {marketValue: > 10000}. Good: WHERE r.marketValue > 10000
- COMMAS: Ensure proper commas in RETURN before aggregate functions (e.g., RETURN a, COUNT(b))
""",

    "fund_profile": """
# FUND PROFILE & NARRATIVE CONTENT
# ⚠️  See UNIVERSAL_RULES at top of file for 16 core Cypher rules

(:Fund {
    ticker,              # Symbol like 'VTI'
    name                 # Full fund name
})

# Profile with prospectus document
(:Fund)-[:DEFINED_BY {year}]->(:Profile {
    summaryProspectus    # Full prospectus text content
})

# === SECTIONS ===
# Embeddings live on Section:Objective AND on child :Chunk nodes for Strategy/Risk.
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

# ⚠️ CRITICAL: Strategy and Risk sections have text: NULL on the Section node.
# Full content lives ONLY in child :Chunk nodes. NEVER use s.text for these sections.
# ALWAYS traverse HAS_CHUNK and return chunk.text.
#
# CORRECT PATTERN — Risk chunks:
#   MATCH (f:Fund {ticker: 'VTI'})-[:DEFINED_BY]->(p:Profile)
#         -[:HAS_SECTION]->(s:Section:Risk)-[:HAS_CHUNK]->(chunk:Chunk)
#   RETURN s.title AS sectionTitle, chunk.text AS chunkContent ORDER BY chunk.id ASC
#
# CORRECT PATTERN — Strategy chunks:
#   MATCH (f:Fund {ticker: 'VTI'})-[:DEFINED_BY]->(p:Profile)
#         -[:HAS_SECTION]->(s:Section:Strategy)-[:HAS_CHUNK]->(chunk:Chunk)
#   RETURN s.title AS sectionTitle, chunk.text AS chunkContent ORDER BY chunk.id ASC
(:Profile)-[:HAS_SECTION]->(:Section:Strategy {
    text: null,          # ⚠️ NULL — do NOT return s.text. Use HAS_CHUNK instead.
    title
})

(:Profile)-[:HAS_SECTION]->(:Section:Risk {
    text: null,          # ⚠️ NULL — do NOT return s.text. Use HAS_CHUNK instead.
    title
})

# === CHUNKS (the actual embedded units — this is where the text lives) ===
(:Section:Strategy)-[:HAS_CHUNK]->(:Chunk {
    text,                # Chunk text — THIS is the actual content, not Section.text
    embedding            # Vector embedding for semantic search
})

(:Section:Risk)-[:HAS_CHUNK]->(:Chunk {
    text,                # Chunk text — THIS is the actual content, not Section.text
    embedding            # Vector embedding for semantic search
})

# === VECTOR INDEXES ===
# NOTE: chunkEmbeddingIndex is the single physical index for ALL :Chunk nodes —
# Profile and 10-K chunks share it. The traversal path determines which you get.
# CALL db.index.vector.queryNodes('chunkEmbeddingIndex', $k, $queryVector)
#   → returns :Chunk nodes from Profile Strategy / Risk sections.
#   Traverse: chunk <-[:HAS_CHUNK]-(section:Section) <-[:HAS_SECTION]-(p:Profile) <-[:DEFINED_BY]-(f:Fund)
#   Filter by section label, e.g.  WHERE 'Strategy' IN labels(section)
#   Example: CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 15, $queryVector) YIELD node AS chunk, score
#            MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:Strategy)<-[:HAS_SECTION]-(p:Profile)<-[:DEFINED_BY]-(f:Fund {ticker: 'VTI'})
#            RETURN chunk.text AS text, chunk.id AS chunkId, f.ticker AS ticker, score ORDER BY score DESC LIMIT 10
#
# CALL db.index.vector.queryNodes('profileObjectiveIndex', $k, $queryVector)
#   → returns :Section:Objective nodes (embedding ON the section).
#   Traverse: objective <-[:HAS_SECTION]-(p:Profile) <-[:DEFINED_BY]-(f:Fund)
#   Example: CALL db.index.vector.queryNodes('profileObjectiveIndex', 15, $queryVector) YIELD node AS objective, score
#            MATCH (objective)<-[:HAS_SECTION]-(p:Profile)<-[:DEFINED_BY]-(f:Fund)
#            RETURN objective.text AS text, objective.id AS chunkId, f.ticker AS ticker, score ORDER BY score DESC LIMIT 10

(:Profile)-[:EXTRACTED_FROM]->(:Document {
    accessionNumber, url, form, filingDate, reportingDate
})

FUND_PROFILE SPECIFIC RULES (beyond UNIVERSAL_RULES):
- ⚠️ Strategy & Risk text is NULL on Section — use HAS_CHUNK to get content from chunks
- MANDATORY: Use vector search for content queries:
  • Strategy/Risk content → chunkEmbeddingIndex
  • Objectives → profileObjectiveIndex
  • Plain MATCH only for summaryProspectus or structural queries (not content)
- NO TEXT FILTERS on vector search (removes results to 0). Section titles are generic ('Risk Factors').
  Bad: YIELD chunk MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:Strategy) WHERE s.title CONTAINS 'keyword'
  Good: YIELD chunk, score MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:Strategy)
- GENERAL QUERIES: If question is broad ("which funds...", "find funds...") WITHOUT specific ticker,
  do NOT add entity filters — search globally across all funds
- ENTITY FILTERS IN VECTOR: Only add {ticker: 'X'} if BOTH: (a) Entity score ≈ 100, AND (b) ticker
  appears in question text. Never use CONTAINS on ticker/name inside vector traversal.
- ALWAYS return ticker and name. Do NOT extract netAssets from these text nodes.
""",

    "company_filing": """
# COMPANY 10-K FILING & BUSINESS INFORMATION
# ⚠️  See UNIVERSAL_RULES at top of file for 16 core Cypher rules

(:Company {
    ticker,              # Stock ticker like 'AAPL', 'MSFT'
    name,                # Company name
    cik                  # SEC Central Index Key
})

# 10-K filing — year is on the REPORTS_IN relationship
(:Company)-[:REPORTS_IN {year}]->(:Filing10K)
# Year property is in the relationship REPORTS_IN not in the Filing10K
(:Filing10K)-[:EXTRACTED_FROM]->(:Document {
    accessionNumber,     # SEC accession number
    url,                 # SEC EDGAR URL
    form,                # Form type (10-K)
    filingDate,          # Filing date
    reportingDate        # Reporting period
})

# === SECTIONS — text storage rules ===
# ⚠️ CRITICAL: RiskFactor, BusinessInformation, LegalProceeding, ManagementDiscussion
#   have text: NULL on the Section node. Full content is in child :Chunk nodes.
#   ALWAYS traverse HAS_CHUNK and return chunk.text for these four section types.
#
# Properties IS the exception — text IS populated directly on the Section node.
#
# CORRECT PATTERN for RiskFactor / BusinessInformation / LegalProceeding / ManagementDiscussion:
#   MATCH (c:Company {ticker: 'AAPL'})-[r:REPORTS_IN]->(f:Filing10K)
#         -[:HAS_SECTION]->(s:Section:RiskFactor)-[:HAS_CHUNK]->(chunk:Chunk)
#   RETURN s.title AS sectionTitle, chunk.text AS chunkContent, r.year AS year
#   ORDER BY r.year DESC, chunk.id ASC
#
# CORRECT PATTERN for Properties (text IS on the node):
#   MATCH (c:Company {ticker: 'MSFT'})-[:REPORTS_IN]->(f:Filing10K)-[:HAS_SECTION]->(s:Section:Properties)
#   RETURN s.title AS sectionTitle, s.text AS properties
#
# IMPORTANT: The MD&A section label is 'ManagementDiscussion' — use this exact spelling.
(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {text: null, title})          # text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {text: null, title}) # text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {text: null, title})     # text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:ManagementDiscussion {text: null, title})# text NULL — use HAS_CHUNK
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {text, title})                # text IS populated

# === CHUNKS (the embedded units — used by chunkEmbeddingIndex) ===
(:Section:RiskFactor)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:BusinessInformation)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:LegalProceeding)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:ManagementDiscussion)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Section:Properties)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})

# === VECTOR INDEX ===
# CALL db.index.vector.queryNodes('chunkEmbeddingIndex', $k, $queryVector)
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
    label, value
})

(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment {
    label, value, percentage
})

COMPANY_FILING SPECIFIC RULES (beyond UNIVERSAL_RULES):
- Fulltext index: companyNameIndex (YIELD node AS c, score)
- ⚠️ RiskFactor, BusinessInformation, LegalProceeding, ManagementDiscussion have NULL text on Section
  • Use HAS_CHUNK to get content from chunks
  • Properties section HAS text populated directly
  • MD&A label is 'ManagementDiscussion' (exact spelling)
- MANDATORY: Use vector search for content queries:
  • Pattern: CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 15, $queryVector)
              YIELD node AS chunk, score
              MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:RiskFactor)<-[:HAS_SECTION]-(f:Filing10K)<-[r:REPORTS_IN]-(c:Company {ticker: 'AAPL'})
              RETURN chunk.text AS text, chunk.id AS chunkId, c.ticker AS ticker, r.year AS filingYear, score ORDER BY score DESC LIMIT 10
- NO TEXT FILTERS on vector search (silently returns 0). Section titles are generic ('Risk Factors').
- GENERAL QUERIES: If broad ("find risks across companies", "which companies disclose...") WITHOUT
  specific company, do NOT add entity filters — search globally
- ENTITY FILTERS: Only add {ticker: 'X'} if BOTH: (a) Entity score ≈ 100, AND (b) ticker in question
- Relationship: HAS_FINANCIALS (NOT HAS_FINACIALS with one 'N')
""",

    "company_people": """
# COMPANY PEOPLE, COMPENSATION & INSIDER TRANSACTIONS
# ⚠️  See UNIVERSAL_RULES at top of file for 16 core Cypher rules

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
    accessionNumber, url, form, filingDate, reportingDate
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
    accessionNumber, url, form, filingDate, reportingDate
})

COMPANY_PEOPLE SPECIFIC RULES (beyond UNIVERSAL_RULES):
- Fund managers: MANAGED_BY on Fund, NOT Company
- Company CEOs/executives: HAS_CEO on Company
- Insider transaction direction: (InsiderTransaction)-[:MADE_BY]->(Person), NOT reverse
- Fulltext indexes:
  • personNameIndex (YIELD node AS person, score)
  • companyNameIndex (YIELD node AS c, score)
- INLINE MATH: NEVER use > or < inside patterns. Bad: {price: > 100}. Good: WHERE price > 100
""",
}


# ═══════════════════════════════════════════════════════════════════════════
# V2 — COMPACT SCHEMA SLICES (~50% fewer tokens, same structural coverage)
# Removes per-property comments and verbose correct-pattern examples.
# Critical warnings and index names are preserved as one-liners.
# ═══════════════════════════════════════════════════════════════════════════
SCHEMA_SLICES_V2 = {
    "not_related": None,

    "fund_basic": """
# FUND BASIC
(:Provider {name})-[:MANAGES]->(:Trust {name})-[:ISSUES]->(:Fund {ticker, name, cik, securityExchange})
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
(:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})
(:Fund)-[r:HAS_FINANCIAL_HIGHLIGHT {year}]->(:FinancialHighlight {turnover, expenseRatio, totalReturn, netAssets, netAssetsValueBeginning, netAssetsValueEnd, netIncomeRatio, advisoryFees})
(:Fund)-[r:HAS_AVERAGE_RETURNS {year}]->(:AverageReturns {return1y, return5y, return10y, returnInception})
(:Fund)-[r:HAS_CHART {year}]->(:Image {category, svg, title})
(:Fund)-[r:HAS_TABLE {year}]->(:Table {content, title})
⚠️ netAssets/expenseRatio/advisoryFees → FinancialHighlight node, NOT Fund | year is on the RELATIONSHIP, not the node | turnover absolute (2=2%) | numberHoldings → Portfolio.count
Indexes: fundNameIndex, providerNameIndex, trustNameIndex
""",

    "fund_portfolio": """
# FUND PORTFOLIO
(:Fund {ticker, name})-[:HAS_PORTFOLIO]->(p:Portfolio {date, seriesId, count})
(p)-[r:HAS_HOLDING {shares, marketValue, weight, fairValueLevel, isRestricted, payoffProfile}]->(h:Holding {name, ticker, isin, lei, country, category, categoryDesc, issuerCategory, businessAddress})
(h)-[:OF_ASSET_TYPE]->(:AssetCategory {code, name, category, subcategory})
(h)-[:REPRESENTS]->(:Company {ticker, name})
(:Fund)-[h:HAS_SECTOR_ALLOCATION {weight, year}]->(:Sector {name})
(:Fund)-[h:HAS_REGION_ALLOCATION {weight, year}]->(:Region {name})
(:Fund)-[r:HAS_TABLE {year}]->(:Table {title, content})
(p)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, type, filingDate, reportingDate})
⚠️ weight/marketValue/payoffProfile → HAS_HOLDING relationship NOT Holding node | count → p.count NOT COUNT(h) | year on HAS_SECTOR_ALLOCATION/HAS_REGION_ALLOCATION
Indexes: holdingNameIndex
""",

    "fund_profile": """
# FUND PROFILE
(:Fund {ticker, name})-[:DEFINED_BY {year}]->(:Profile {summaryProspectus})
(:Profile)-[:HAS_SECTION]->(:Section:Objective {text, title, embedding})
(:Profile)-[:HAS_SECTION]->(:Section:PerformanceCommentary {text, title})
(:Profile)-[:HAS_SECTION]->(:Section:Risk {text: null, title})
(:Profile)-[:HAS_SECTION]->(:Section:Strategy {text: null, title})
(:Section:Risk)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Section:Strategy)-[:HAS_CHUNK]->(:Chunk {text, embedding})
(:Profile)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, form, filingDate, reportingDate})
⚠️ Risk/Strategy: text=NULL on Section → ALWAYS use HAS_CHUNK | Objective: embedding on Section (no chunks)
Vector indexes: chunkEmbeddingIndex (chunks), profileObjectiveIndex (objectives)
Structural: MATCH (f:Fund {ticker:'X'})-[:DEFINED_BY]->(p:Profile)-[:HAS_SECTION]->(s:Section:Risk)-[:HAS_CHUNK]->(c:Chunk) RETURN c.text
Vector: CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 15, $queryVector) YIELD node AS chunk, score MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:Strategy)<-[:HAS_SECTION]-(p:Profile)<-[:DEFINED_BY]-(f:Fund {ticker:'X'}) RETURN chunk.text AS text, chunk.id AS chunkId, f.ticker AS ticker, score ORDER BY score DESC LIMIT 10
""",

    "company_filing": """
# COMPANY 10-K FILING
(:Company {ticker, name, cik})-[r:REPORTS_IN {year}]->(:Filing10K)
# Year property is in the relationship REPORTS_IN not in the Filing10K
(:Filing10K)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, form, filingDate, reportingDate})
(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {text: null, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {text: null, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {text: null, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:ManagementDiscussion {text: null, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {text, title})
(:Section)-[:HAS_CHUNK]->(:Chunk {text, embedding, title})
(:Filing10K)-[:HAS_FINANCIALS]->(:Section:Financials {incomeStatement, balanceSheet, cashFlow, fiscalYear})
(:Section:Financials)-[:HAS_METRIC]->(:FinancialMetric {label, value})-[:HAS_SEGMENT]->(:Segment {label, value, percentage})
⚠️ RiskFactor/BusinessInformation/LegalProceeding/ManagementDiscussion: text=NULL → use HAS_CHUNK | Properties: text IS populated | MD&A label = 'ManagementDiscussion'
Vector index: chunkEmbeddingIndex | Fulltext: companyNameIndex
Structural: MATCH (c:Company {ticker:'X'})-[r:REPORTS_IN]->(f:Filing10K)-[:HAS_SECTION]->(s:Section:RiskFactor)-[:HAS_CHUNK]->(ch:Chunk) RETURN ch.text AS text, r.year AS year
Vector: CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 15, $queryVector) YIELD node AS chunk, score MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:RiskFactor)<-[:HAS_SECTION]-(f:Filing10K)<-[r:REPORTS_IN]-(c:Company {ticker:'X'}) RETURN chunk.text AS text, chunk.id AS chunkId, c.ticker AS ticker, r.year AS filingYear, score ORDER BY score DESC LIMIT 10
""",

    "company_people": """
# COMPANY PEOPLE & INSIDER TRANSACTIONS
(:Fund {ticker, name})-[:MANAGED_BY {year}]->(:Person {name})
(:Company {ticker, name, cik})-[r:HAS_CEO {ceoCompensation, ceoActuallyPaid, date}]->(:Person {name})
(:Person)-[:RECEIVED_COMPENSATION]->(:CompensationPackage {totalCompensation, shareholderReturn, date})
(:CompensationPackage)-[:AWARDED_BY]->(:Company)
(:CompensationPackage)-[:DISCLOSED_IN]->(:Document {accessionNumber, url, form, filingDate, reportingDate})
(:Company)-[:HAS_INSIDER_TRANSACTION]->(:InsiderTransaction {transactionDate, position, transactionType, shares, price, value, remainingShares})
(:InsiderTransaction)-[:MADE_BY]->(:Person {name})
(:InsiderTransaction)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, form, filingDate, reportingDate})
⚠️ Fund managers → MANAGED_BY on Fund | Company executives → HAS_CEO on Company
Indexes: personNameIndex, companyNameIndex
HAS_INSIDER_TRANSACTION is a relationship, not a node do not add a name to it, the properties are in the InsiderTransaction node.
⚠️ BAD: [cp:RECEIVED_COMPENSATION]->(cp:CompensationPackage) | GOOD: -[:RECEIVED_COMPENSATION]->(cp:CompensationPackage)

""",
}

# ═══════════════════════════════════════════════════════════════════════════
# VERSION REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
SCHEMA_VERSIONS: dict[str, dict] = {
    "v1": SCHEMA_SLICES_V1,   # verbose — full comments, correct-pattern examples
    "v2": SCHEMA_SLICES_V2,   # compact — ~50% fewer tokens, same structural coverage
}

DEFAULT_SCHEMA_VERSION = "v1"


def get_schema_for_category(category: str, version: str = DEFAULT_SCHEMA_VERSION) -> str:
    """Get the schema slice for a given query category."""
    slices = SCHEMA_VERSIONS.get(version, SCHEMA_SLICES_V1)
    schema = slices.get(category, "")
    return schema if schema else ""


def get_merged_schema(categories: list, version: str = DEFAULT_SCHEMA_VERSION) -> str:
    """
    Merge multiple category schemas into a single schema string.
    Used when a query spans multiple categories.
    """
    slices = SCHEMA_VERSIONS.get(version, SCHEMA_SLICES_V1)
    parts = []
    for category in categories:
        if category == "not_related":
            continue
        schema = slices.get(category, "")
        if schema:
            parts.append(f"=== Schema: {category} ===\n{schema}")
    return "\n\n".join(parts)
