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
    cik,                 # SEC Central Index Key
    securityExchange     # Exchange like 'NASDAQ', 'NYSE'
})
# ⚠️ netAssets, expenseRatio, advisoryFees are on FinancialHighlight, NOT on Fund
(:Fund)-[:HAS_SHARE_CLASS]->(:ShareClass {name, description})
(:Fund)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, form, filingDate, reportingDate})

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

QUERY RULES:
- For fund/provider/trust name searches use fulltext indexes BEFORE any MATCH:
  CALL db.index.fulltext.queryNodes('fundNameIndex', 'search_term') YIELD node AS f, score RETURN f.ticker, f.name ORDER BY score DESC LIMIT 5
  CALL db.index.fulltext.queryNodes('providerNameIndex', 'search_term') YIELD node AS p, score RETURN p.name ORDER BY score DESC LIMIT 5
  CALL db.index.fulltext.queryNodes('trustNameIndex', 'search_term') YIELD node AS t, score RETURN t.name ORDER BY score DESC LIMIT 5
- ALWAYS alias the YIELD node: `YIELD node AS f` — do NOT use `node` directly or invent variable names like `fund`/`trust`/`provider`.
- For exact ticker matching use {ticker: 'VTI'} directly (no index needed).
- numberHoldings → use Portfolio.count: MATCH (f)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.count
- The year is on the RELATIONSHIP [r:HAS_FINANCIAL_HIGHLIGHT {year}], not the node: use r.year.
- turnover is absolute (2 = 2%, not 0.02).
- Always include the relationship variable (e.g., [r:HAS_FINANCIAL_HIGHLIGHT]) and use it in RETURN.
- ALWAYS return the source document via the EXTRACTED_FROM relationship when it exists.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, no markdown formatting, no ```cypher blocks.
2. UNIQUE VARIABLES (FATAL ERROR PREVENTION): NEVER use the same variable name for both a relationship and a node. For example, `-[p:HAS_PORTFOLIO]->(p:Portfolio)` is invalid and will crash. Always use distinct names like `-[rel:HAS_PORTFOLIO]->(p:Portfolio)`.
3. SUBQUERY SYNTAX: Do NOT use SQL-style subqueries like `ticker IN (MATCH...)`. To filter based on a sub-pattern, use `WHERE EXISTS { MATCH ... }` or use a `WITH` clause.
4. AGGREGATIONS & GROUPING: If the question asks for a global calculation (e.g., highest, average, total across all nodes), DO NOT include row-specific identifiers (like ticker, name, or year) in the RETURN clause. Doing so triggers an implicit GROUP BY and ruins the calculation.
5. EXTRA PROPERTIES: For standard lists/queries (NOT aggregations), always return extra identifying properties (ticker, name) and relationship properties beyond what was asked.
6. SCHEMA STRICTNESS: Use property names EXACTLY as they appear in the provided schema. Do not hallucinate properties like `.text` if the schema specifies `.summaryProspectus`.
7. FILTERING: Use CONTAINS or regex for text searches; use >, <, =, etc., for numeric comparisons. WHERE clauses must follow MATCH or WITH, never RETURN.
8. DUPLICATES: If a MATCH traverses nodes not included in the RETURN, use RETURN DISTINCT.
9. CONTEXT MATCHING: If entity names appear in Entity Context, use them EXACTLY as written.
10. FEW-SHOT REPLICATION: If an example is marked [★ VERY SIMILAR], strictly replicate its structural logic — only swap the specific entity names or target properties.
11. FULLTEXT INDEXES: If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.
12. AVERAGE RETURNS: Use predefined properties like return1y, return5y, return10y instead of calculating based on dates.
13. LATEST: When asked for the latest ALWAYS use ORDER BY property.date DESC LIMIT 1 rather than building complex NOT EXISTS subqueries.
14. TICKER: When the query contains a ticker ALWAYS use that exact ticker.
15. STRICT SCHEMA ALIGNMENT: `netAssets`, `expenseRatio`, and `advisoryFees` are on `FinancialHighlight`, NOT on `Fund`. Access them via the HAS_FINANCIAL_HIGHLIGHT relationship.
16. COMPARING ENTITIES: When asked to compare multiple funds (e.g., "Compare VTI and VOO"), return each fund as a separate row using `IN` (e.g., `WHERE f.ticker IN ['VTI', 'VOO'] RETURN f.ticker...`).
17. HISTORICAL GROWTH: To calculate growth over N years, MATCH all highlights, ORDER BY year, COLLECT them, then compare `highlights[0]` (latest) with a prior index (e.g., `highlights[N]`).
18. DIVISION BY ZERO: When calculating percentages, ALWAYS use `CASE WHEN denominator = 0 THEN 0 ELSE (numerator * 100.0 / denominator) END`.
19. INCOMPLETE FILTERS: NEVER generate empty or incomplete property filters like `(n:Label {name})`. Only include explicit values like `{name: 'Vanguard'}`.
20. NO YEAR FILTER UNLESS ASKED: Do NOT add year/date filters (e.g. {{year: 2023}} or WHERE r.year = ...) unless the question explicitly mentions a specific year or date range.
""",

    "fund_portfolio": """
# FUND PORTFOLIO COMPOSITION & ALLOCATIONS
# Scope: holdings, weights, sector/regional breakdowns, asset categories

(:Fund {
    ticker,              # Symbol like 'VTI'
    name,                # Full fund name
    securityExchange     # Exchange like 'NASDAQ', 'NYSE'
})
# ⚠️ numberHoldings is NOT on Fund. Use Portfolio.count:
# MATCH (f:Fund)-[:HAS_PORTFOLIO]->(p:Portfolio) RETURN p.count AS numberHoldings

# Portfolio structure
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

# Sector allocations (year on the relationship)
(:Fund)-[h:HAS_SECTOR_ALLOCATION {weight, year}]->(:Sector {
    name                 # Sector name (Technology, Healthcare, etc.)
})

# Regional/geographic allocations (year on the relationship)
(:Fund)-[h:HAS_REGION_ALLOCATION {weight, year}]->(:Region {
    name                 # Region name (North America, Asia-Pacific, etc.)
})

# Link holdings to companies
(h)-[:REPRESENTS]->(:Company {ticker, name})

(p)-[:EXTRACTED_FROM]->(:Document {accessionNumber, url, form, filingDate, reportingDate})

QUERY RULES:
- Use p.count for number of holdings — do NOT count holdings manually with COUNT(h).
- weight, marketValue, payoffProfile are on the [r:HAS_HOLDING] RELATIONSHIP, not the Holding node.
- year is on the HAS_SECTOR_ALLOCATION and HAS_REGION_ALLOCATION relationships, use h.year.
- For holding name/ticker searches use: CALL db.index.fulltext.queryNodes('holdingNameIndex', 'search_term') YIELD node AS h, score — ALWAYS alias YIELD node.
- ALWAYS return the source document via EXTRACTED_FROM when it exists.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, no markdown formatting, no ```cypher blocks.
2. UNIQUE VARIABLES (FATAL ERROR PREVENTION): NEVER use the same variable name for both a relationship and a node. For example, `-[p:HAS_PORTFOLIO]->(p:Portfolio)` is invalid and will crash. Always use distinct names like `-[rel:HAS_PORTFOLIO]->(p:Portfolio)`.
3. SUBQUERY SYNTAX: Do NOT use SQL-style subqueries like `ticker IN (MATCH...)`. To filter based on a sub-pattern, use `WHERE EXISTS { MATCH ... }` or use a `WITH` clause.
4. AGGREGATIONS & GROUPING: If the question asks for a global calculation (e.g., highest, average, total across all nodes), DO NOT include row-specific identifiers (like ticker, name, or year) in the RETURN clause. Doing so triggers an implicit GROUP BY and ruins the calculation.
5. EXTRA PROPERTIES: For standard lists/queries (NOT aggregations), always return extra identifying properties (ticker, name) and relationship properties beyond what was asked.
6. SCHEMA STRICTNESS: Use property names EXACTLY as they appear in the provided schema. Do not hallucinate properties like `.text` if the schema specifies `.summaryProspectus`.
7. FILTERING: Use CONTAINS or regex for text searches; use >, <, =, etc., for numeric comparisons. WHERE clauses must follow MATCH or WITH, never RETURN.
8. DUPLICATES: If a MATCH traverses nodes not included in the RETURN, use RETURN DISTINCT.
9. CONTEXT MATCHING: If entity names appear in Entity Context, use them EXACTLY as written.
10. FEW-SHOT REPLICATION: If an example is marked [★ VERY SIMILAR], strictly replicate its structural logic — only swap the specific entity names or target properties.
11. FULLTEXT INDEXES: If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.
12. LATEST: When asked for the latest ALWAYS use ORDER BY property.date DESC LIMIT 1 rather than building complex NOT EXISTS subqueries.
13. TICKER: When the query contains a ticker ALWAYS use that exact ticker.
14. STRICT SCHEMA ALIGNMENT: `payoffProfile`, `marketValue`, and `weight` are properties of the `[r:HAS_HOLDING]` relationship, NOT the `Holding` node.
15. INLINE MATH: NEVER use math operators (`>`, `<`) inside curly braces in MATCH patterns (BAD: `[r:HAS_HOLDING {marketValue: > 10000}]`). ALWAYS use a `WHERE` clause (GOOD: `WHERE r.marketValue > 10000`).
16. DIVISION BY ZERO: When calculating percentage ratios, ALWAYS use `CASE WHEN total = 0 THEN 0 ELSE (part * 100.0 / total) END`.
17. COMMAS: Ensure proper commas are placed between variables in the `RETURN` statement before aggregate functions (e.g., `RETURN a, COUNT(b)`).
18. INCOMPLETE FILTERS: NEVER generate empty or incomplete property filters like `(n:Label {name})`. Only include explicit values like `{name: 'Vanguard'}`.
19. NO YEAR FILTER UNLESS ASKED: Do NOT add year/date filters (e.g. {{year: 2023}} or WHERE r.year = ...) unless the question explicitly mentions a specific year or date range.
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
# NOTE: chunkEmbeddingIndex is the single physical index for ALL :Chunk nodes —
# Profile and 10-K chunks share it. The traversal path determines which you get.
# CALL db.index.vector.queryNodes('chunkEmbeddingIndex', $k, $queryVector)
#   → returns :Chunk nodes from Profile Strategy / RiskFactor sections.
#   Traverse: chunk <-[:HAS_CHUNK]-(section:Section) <-[:HAS_SECTION]-(p:Profile) <-[:DEFINED_BY]-(f:Fund)
#   Filter by section label, e.g.  WHERE 'Strategy' IN labels(section)
#
# CALL db.index.vector.queryNodes('profileObjectiveIndex', $k, $queryVector)
#   → returns :Section:Objective nodes (embedding ON the section).
#   Traverse: objective <-[:HAS_SECTION]-(p:Profile) <-[:DEFINED_BY]-(f:Fund)

(:Profile)-[:EXTRACTED_FROM]->(:Document {
    accessionNumber,     # SEC accession number
    url,                 # SEC EDGAR URL
    form,                # Form type (N-CSR, etc.)
    filingDate,          # Filing date
    reportingDate        # Reporting date
})

QUERY RULES:
- MANDATORY: For any question about the CONTENT of a fund's strategy, risks, or objectives
  (e.g., "what risks", "describe the strategy", "which funds focus on..."), you MUST use
  a vector index — NOT a plain MATCH pattern.
  - Questions about strategy/risk content → chunkEmbeddingIndex
  - Questions about investment objectives → profileObjectiveIndex
- Plain MATCH is ONLY allowed when retrieving the full summaryProspectus text from Profile
  or when filtering by year/structure (not by content meaning).
- NEVER access Section.text on Strategy or RiskFactor sections — those sections store content
  in Chunk nodes only. Always traverse via [:HAS_CHUNK] and use chunkEmbeddingIndex.
- NO TEXT FILTERS ON VECTOR SEARCH: When using a vector index (chunkEmbeddingIndex,
  profileObjectiveIndex), NEVER add WHERE CONTAINS or WHERE text CONTAINS filters on the
  Section or Chunk nodes. The vector embedding already handles semantic matching — adding a
  text filter on top will return 0 results because Section.text is null on these nodes.
  BAD:  ... YIELD node AS chunk, score MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:Strategy) WHERE s.text CONTAINS 'small-cap'
  GOOD: ... YIELD node AS chunk, score MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:Strategy)
- GENERAL QUERIES (no specific entity): If the question uses broad language ("which funds...",
  "find funds...", "show me funds that...") WITHOUT naming a specific ticker or fund name,
  do NOT add entity filters ({ticker: ...}, WHERE f.ticker = ...) to the vector search.
  The entity context provided may have matched spuriously — ignore it and search globally.
- ALWAYS return ticker and name alongside the requested data.
- Do NOT attempt to extract netAssets or numerical performance from these text nodes.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, no markdown formatting, no ```cypher blocks.
2. UNIQUE VARIABLES (FATAL ERROR PREVENTION): NEVER use the same variable name for both a relationship and a node. Always use distinct names like `-[rel:HAS_SECTION]->(s:Section)`.
3. SUBQUERY SYNTAX: Do NOT use SQL-style subqueries like `ticker IN (MATCH...)`. Use `WHERE EXISTS { MATCH ... }` or a `WITH` clause.
4. SCHEMA STRICTNESS: Use property names EXACTLY as they appear in the provided schema. Do not use `.text` if the schema specifies `.summaryProspectus`.
5. FILTERING: WHERE clauses must follow MATCH or WITH, never RETURN.
6. DUPLICATES: If a MATCH traverses nodes not included in the RETURN, use RETURN DISTINCT.
7. CONTEXT MATCHING: If entity names appear in Entity Context, use them EXACTLY as written.
8. FEW-SHOT REPLICATION: If an example is marked [★ VERY SIMILAR], strictly replicate its structural logic — only swap the specific entity names or target properties.
9. TICKER: When the query contains a ticker ALWAYS use that exact ticker.
10. LATEST: When asked for the latest profile/year, use ORDER BY year DESC LIMIT 1.
11. NO YEAR FILTER UNLESS ASKED: Do NOT add year/date filters (e.g. {{year: 2023}} or WHERE r.year = ...) unless the question explicitly mentions a specific year or date range.
12. NO TEXT FILTERS ON VECTOR SEARCH: When using chunkEmbeddingIndex or profileObjectiveIndex, NEVER add WHERE ... CONTAINS on Section or Chunk nodes. The vector handles semantic matching — a text filter returns 0 because Section.text is null. BAD: `WHERE s.text CONTAINS 'keyword'` after a vector YIELD. GOOD: remove it entirely.
13. GENERAL QUERIES (no entity): If the question asks broadly ("which funds...", "find funds that...") WITHOUT a specific ticker or name, do NOT add entity filters. Search globally.
""",

    "company_filing": """
# COMPANY 10-K FILING & BUSINESS INFORMATION
# Scope: Business financials, income statements, risk factors, MD&A sections.
# Embeddings live on :Chunk nodes (NOT on Section nodes). Use chunkEmbeddingIndex.

(:Company {
    ticker,              # Stock ticker like 'AAPL', 'MSFT'
    name,                # Company name
    cik                  # SEC Central Index Key
})

# 10-K filing — year is on the REPORTS_IN relationship
(:Company)-[:REPORTS_IN {year}]->(:Filing10K)

(:Filing10K)-[:EXTRACTED_FROM]->(:Document {
    accessionNumber,     # SEC accession number
    url,                 # SEC EDGAR URL
    form,                # Form type (10-K)
    filingDate,          # Filing date
    reportingDate        # Reporting period
})

# === SECTIONS (text + title only — NO embeddings here) ===
# IMPORTANT: The MD&A section label is 'ManagementDiscussion' — use this exact spelling.
(:Filing10K)-[:HAS_SECTION]->(:Section:RiskFactor {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:BusinessInformation {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:LegalProceeding {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:ManagementDiscussion {text, title})
(:Filing10K)-[:HAS_SECTION]->(:Section:Properties {text, title})

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
    label,               # Metric name (Revenue, NetIncome, etc.)
    value                # Metric value
})

(:FinancialMetric)-[:HAS_SEGMENT]->(:Segment {
    label,               # Segment name
    value,               # Segment value
    percentage           # Percentage of total
})

QUERY RULES:
- For company name searches use: CALL db.index.fulltext.queryNodes('companyNameIndex', 'search_term') YIELD node AS c, score — ALWAYS alias YIELD node.
- MANDATORY: For any question about the CONTENT of a 10-K section (e.g., "what does Apple say
  about...", "describe how Microsoft...", "find risk factors related to..."), you MUST use
  chunkEmbeddingIndex — NOT a plain MATCH pattern.
  Pattern: CALL db.index.vector.queryNodes('chunkEmbeddingIndex', $k, $queryVector)
           YIELD node AS chunk, score
           MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:SectionLabel)<-[:HAS_SECTION]-(f:Filing10K)<-[r:REPORTS_IN]-(c:Company)
  Filter by company ticker or section label as needed.
- Plain MATCH on Section.text is ONLY allowed when retrieving full section text by exact structural
  filter (e.g., retrieve the entire Properties section text, not a semantic search within it).
- NO TEXT FILTERS ON VECTOR SEARCH: When using chunkEmbeddingIndex, NEVER add WHERE CONTAINS
  filters on Section or Chunk nodes. The vector embedding handles semantic matching — a text
  filter on top will silently return 0 results.
  BAD:  ... YIELD node AS chunk, score MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:RiskFactor) WHERE s.text CONTAINS 'interest rate'
  GOOD: ... YIELD node AS chunk, score MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:RiskFactor)
- GENERAL QUERIES (no specific entity): If the question uses broad language ("find risk factors
  across all companies", "which companies disclose...") WITHOUT naming a specific company ticker
  or name, do NOT add entity filters to the vector search. Ignore any resolved entity context
  and search globally across all companies.
- Company ticker is a stock ticker like 'AAPL', 'MSFT'.
- The MD&A section label is 'ManagementDiscussion' — use this exact spelling.
- The financials relationship is HAS_FINANCIALS (also exists as HAS_FINACIALS with one 'N' — use HAS_FINANCIALS).

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, no markdown formatting, no ```cypher blocks.
2. UNIQUE VARIABLES (FATAL ERROR PREVENTION): NEVER use the same variable name for both a relationship and a node. For example, `-[p:HAS_PORTFOLIO]->(p:Portfolio)` is invalid and will crash. Always use distinct names like `-[rel:HAS_PORTFOLIO]->(p:Portfolio)`.
3. SUBQUERY SYNTAX: Do NOT use SQL-style subqueries like `ticker IN (MATCH...)`. To filter based on a sub-pattern, use `WHERE EXISTS { MATCH ... }` or use a `WITH` clause.
4. AGGREGATIONS & GROUPING: If the question asks for a global calculation (e.g., highest, average, total across all nodes), DO NOT include row-specific identifiers (like ticker, name, or year) in the RETURN clause. Doing so triggers an implicit GROUP BY and ruins the calculation.
5. EXTRA PROPERTIES: For standard lists/queries (NOT aggregations), always return extra identifying properties (ticker, name) and relationship properties beyond what was asked.
6. SCHEMA STRICTNESS: Use property names EXACTLY as they appear in the provided schema. Do not hallucinate properties like `.text` if the schema specifies `.summaryProspectus`.
7. FILTERING: Use CONTAINS or regex for text searches; use >, <, =, etc., for numeric comparisons. WHERE clauses must follow MATCH or WITH, never RETURN.
8. DUPLICATES: If a MATCH traverses nodes not included in the RETURN, use RETURN DISTINCT.
9. CONTEXT MATCHING: If entity names appear in Entity Context, use them EXACTLY as written.
10. FEW-SHOT REPLICATION: If an example is marked [★ VERY SIMILAR], strictly replicate its structural logic — only swap the specific entity names or target properties.
11. FULLTEXT INDEXES: If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.
12. LATEST: When asked for the latest ALWAYS use ORDER BY property.date DESC LIMIT 1 rather than building complex NOT EXISTS subqueries.
13. TICKER: When the query contains a ticker ALWAYS use that exact ticker.
14. NO YEAR FILTER UNLESS ASKED: Do NOT add year/date filters (e.g. {{year: 2023}} or WHERE r.year = ...) unless the question explicitly mentions a specific year or date range.
15. NO TEXT FILTERS ON VECTOR SEARCH: When using chunkEmbeddingIndex, NEVER add WHERE ... CONTAINS on Section or Chunk nodes after the vector YIELD. The vector handles semantic matching — a text filter returns 0 because Section.text is null. BAD: `WHERE s.text CONTAINS 'keyword'` after a vector YIELD. GOOD: remove it entirely.
16. GENERAL QUERIES (no entity): If the question asks broadly ("which companies...", "find companies that...") WITHOUT a specific ticker or company name, do NOT add entity filters. Search globally.
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

QUERY RULES:
- Fund managers are linked via MANAGED_BY (on Fund, not Company).
- Company CEOs and executives are linked via HAS_CEO (on Company).
- MADE_BY direction: (InsiderTransaction)-[:MADE_BY]->(Person) — NOT the reverse.
- For person name searches: CALL db.index.fulltext.queryNodes('personNameIndex', 'search_term') YIELD node AS person, score — ALWAYS alias YIELD node.
- For company name searches: CALL db.index.fulltext.queryNodes('companyNameIndex', 'search_term') YIELD node AS c, score — ALWAYS alias YIELD node.

CRITICAL CYPHER SYNTAX & LOGIC RULES:
1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, no markdown formatting, no ```cypher blocks.
2. UNIQUE VARIABLES (FATAL ERROR PREVENTION): NEVER use the same variable name for both a relationship and a node. For example, `-[p:HAS_PORTFOLIO]->(p:Portfolio)` is invalid and will crash. Always use distinct names like `-[rel:HAS_PORTFOLIO]->(p:Portfolio)`.
3. SUBQUERY SYNTAX: Do NOT use SQL-style subqueries like `ticker IN (MATCH...)`. To filter based on a sub-pattern, use `WHERE EXISTS { MATCH ... }` or use a `WITH` clause.
4. AGGREGATIONS & GROUPING: If the question asks for a global calculation (e.g., highest, average, total across all nodes), DO NOT include row-specific identifiers (like ticker, name, or year) in the RETURN clause. Doing so triggers an implicit GROUP BY and ruins the calculation.
5. EXTRA PROPERTIES: For standard lists/queries (NOT aggregations), always return extra identifying properties (ticker, name) and relationship properties beyond what was asked.
6. SCHEMA STRICTNESS: Use property names EXACTLY as they appear in the provided schema. Do not hallucinate properties like `.text` if the schema specifies `.summaryProspectus`.
7. FILTERING: Use CONTAINS or regex for text searches; use >, <, =, etc., for numeric comparisons. WHERE clauses must follow MATCH or WITH, never RETURN.
8. DUPLICATES: If a MATCH traverses nodes not included in the RETURN, use RETURN DISTINCT.
9. CONTEXT MATCHING: If entity names appear in Entity Context, use them EXACTLY as written.
10. FEW-SHOT REPLICATION: If an example is marked [★ VERY SIMILAR], strictly replicate its structural logic — only swap the specific entity names or target properties.
11. FULLTEXT INDEXES: If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.
12. LATEST: When asked for the latest ALWAYS use ORDER BY property.date DESC LIMIT 1 rather than building complex NOT EXISTS subqueries.
13. TICKER: When the query contains a ticker ALWAYS use that exact ticker.
14. INLINE MATH: NEVER use math operators (`>`, `<`) inside node or relationship patterns. Use `WHERE` clauses instead.
15. NO YEAR FILTER UNLESS ASKED: Do NOT add year/date filters (e.g. {{year: 2023}} or WHERE r.year = ...) unless the question explicitly mentions a specific year or date range.
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
