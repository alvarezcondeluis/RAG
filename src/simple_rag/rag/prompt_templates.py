"""
Prompt Templates for Text2Cypher Translation.

This module contains the LLM prompt templates used for generating and
correcting Cypher queries from natural language.
"""

CYPHER_GENERATION_TEMPLATE = """You are a Neo4j Cypher expert. Convert the question below to a valid Cypher query using the schema strictly and follow the INSTRUCTIONS thoroughly.

Schema:
{schema}

{entity_context}

CRITICAL RULES:
1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, no markdown formatting, no ```cypher blocks.
2. UNIQUE VARIABLES: NEVER use the same variable name for both a relationship and a node (e.g., -[it:HAS_INSIDER_TRANSACTION]->(it:InsiderTransaction) is FATAL). To prevent this, ALWAYS leave relationship arrows anonymous like -[:HAS_INSIDER_TRANSACTION]-> unless you explicitly need a relationship property.
3. SUBQUERY SYNTAX: Do NOT use SQL-style subqueries like `ticker IN (MATCH...)`. To filter based on a sub-pattern, use `WHERE EXISTS {{ MATCH ... }}` or use a `WITH` clause.
4. AGGREGATIONS & GROUPING: If the question asks for a global calculation (e.g., highest, average, total across all nodes), DO NOT include row-specific identifiers (like ticker, name, or year) in the RETURN clause. Doing so triggers an implicit GROUP BY and ruins the calculation.
5. EXTRA PROPERTIES: For standard lists/queries (NOT aggregations), always return extra identifying properties (ticker, name) and relationship properties beyond what was asked.
6. SCHEMA STRICTNESS: Use property names EXACTLY as they appear in the provided schema. Do not hallucinate properties like `.text` if the schema specifies `.summaryProspectus`.
7. FILTERING: Take into account the score of the entity resolver and use it to filter the results by ticker or name.
8. DUPLICATES: If a MATCH traverses nodes not included in the RETURN, use RETURN DISTINCT.
10. FULLTEXT INDEXES: If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.
11. AVERAGE RETURNS: Use predefined properties like return1y, return5y, return10y instead of calculating based on dates.
12. LATEST: When asked for the latest ALWAYS use ORDER BY property.date DESC LIMIT 1 rather than building complex NOT EXISTS subqueries
13. TICKER: Use the exact ticker from the question.
14. NO YEAR FILTER UNLESS ASKED: Only add year/date filters if the question explicitly mentions a specific year or date range.
14b. FILING10K HAS NO YEAR PROPERTY: The `Filing10K` node does NOT have a `year` property. The filing year is stored ONLY on the `REPORTS_IN` relationship. NEVER write `Filing10K {{year: X}}` — it will always return 0 results. ALWAYS use `(f:Filing10K)<-[r:REPORTS_IN]-(c:Company) WHERE r.year = X` instead.
15. NO TEXT FILTERS ON VECTOR SEARCH: Never add WHERE CONTAINS or title filters on Section/Chunk nodes in vector searches. Embeddings already rank by relevance. Text filters silently return 0 results (section titles like 'Risk Factors' won't match topic phrases).
    BAD: ... WHERE r.year = 2025 AND s.title CONTAINS 'iPhone'   ← title filter kills all results
    GOOD: ... WHERE r.year = 2025                                 ← year filter only is fine
16. GENERAL QUERIES: Questions asking broadly ("which funds...", "find funds...") without naming a specific entity → search globally, omit entity filters.
17. ENTITY FILTERS IN VECTOR SEARCH: Add entity filter after YIELD only if BOTH: (a) Entity Context score ≈ 100 (exact match) AND (b) ticker/entity explicitly stated in question text. Never use CONTAINS on entity properties.
18. NULL FILTERING IN NUMERIC QUERIES: For queries asking for greatest/highest/maximum values, or with numeric filtering/ordering, ALWAYS add WHERE clauses to exclude NULL values. Examples: `WHERE fh.expenseRatio IS NOT NULL`, `WHERE fh.expenseRatio > 0`, `WHERE ar.return1y IS NOT NULL`. Apply this ONLY to the property being queried/sorted — do NOT add `fh.expenseRatio > 0` when the question asks about totalReturn or netIncomeRatio.
19. EXAMPLES: If the examples retrieved return something different that what is asked, do not use them as a base for the query.
20. EXAMPLES OVERRIDE SCHEMA FOR NODE TYPE: If the VERY SIMILAR examples (≥90% match) consistently use a different starting node type than the schema (e.g., examples use :Fund but schema shows :Company/:Filing10K), trust the examples — the query router may have misclassified the question. Copy the example's traversal path exactly and substitute only the entity values (ticker, name). Do NOT force-fit the schema's entity type when VERY SIMILAR examples point to a different node.
21. SECTION RETRIEVAL vs. VECTOR SEARCH — choose based on what the question is asking:
    • Use MATCH + s.text (Properties) or MATCH + HAS_CHUNK (RiskFactor, BusinessInformation, etc.) when the question asks to READ or SHOW the WHOLE section: "show me the properties section", "what does the business section say", "show the legal proceedings". These are structural queries that retrieve section content directly.
    • Use CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 10, $queryVector) when the question asks about a SPECIFIC TOPIC, CONCEPT, or DETAIL inside a section: "what AI models does Google have", "what risks does Apple mention about tariffs", "how does Amazon describe its cloud business". These are semantic queries — the vector index finds the relevant chunks by meaning, not by structure. NEVER use MATCH to answer a semantic/topic question about section content.
22. VECTOR RETURN — ALWAYS include `chunk.id AS chunkId` in the RETURN clause of every vector search query:
    CORRECT: RETURN chunk.text AS text, chunk.id AS chunkId, c.ticker AS ticker, r.year AS filingYear, score ORDER BY score DESC LIMIT 10
    Without chunkId the result is treated as empty even when records are returned.
23. VECTOR TRAVERSAL DIRECTION: After a vector YIELD, ALWAYS traverse REPORTS_IN as `(f:Filing10K)<-[r:REPORTS_IN]-(c:Company)` — the arrow points FROM Company TO Filing10K. NEVER write `(f:Filing10K)-[r:REPORTS_IN]->(c:Company)` — it returns 0 results.
    CORRECT: CALL db.index.vector.queryNodes('chunkEmbeddingIndex', 10, $queryVector) YIELD node AS chunk, score
             MATCH (chunk)<-[:HAS_CHUNK]-(s:Section:RiskFactor)<-[:HAS_SECTION]-(f:Filing10K)<-[r:REPORTS_IN]-(c:Company {{ticker: 'AAPL'}})
             RETURN chunk.text AS text, chunk.id AS chunkId, r.year AS filingYear, score ORDER BY score DESC LIMIT 10
24. VECTOR PRE-FILTERING: Do NOT pre-filter with a separate MATCH before the CALL. Put all entity filters in the MATCH that comes AFTER the YIELD.
    BAD: MATCH (c:Company {{ticker:'GOOG'}})-[:REPORTS_IN]->(f:Filing10K) WITH c,f LIMIT 1 CALL db.index.vector...
    GOOD: CALL db.index.vector... YIELD node AS chunk, score MATCH (chunk)<-[:HAS_CHUNK]-...<-[r:REPORTS_IN]-(c:Company {{ticker:'GOOG'}})
25. DATE LITERALS: Neo4j stores dates as Date objects, NOT strings. When filtering by a specific date value, ALWAYS wrap the literal with date('YYYY-MM-DD'). Never compare a date property to a plain string.
    BAD:  WHERE it.transactionDate = '2025-10-29'
    GOOD: WHERE it.transactionDate = date('2025-10-29')
    This applies to ALL date properties: transactionDate, filingDate, reportingDate, date, etc.

Examples:
{examples}

Question: {question}

Cypher Query:"""

CYPHER_GENERATION_TEMPLATE2 = """You are a Neo4j Cypher expert. Convert the question below to a valid Cypher query using the schema strictly.

Schema:
{schema}

Rules:
1. Output ONLY the Cypher query — no explanations, no markdown
2. Use MATCH, WHERE (only after MATCH, never after RETURN), RETURN, COUNT(*) with correct syntax
3. Use property names exactly as in the schema
4. For text search use CONTAINS or regex; for numeric comparisons use >, <, =, etc.
5. Always return extra identifying properties (ticker, name, etc.) beyond what was asked
6. Always return relationship properties
7. If entity names appear in Entity Context, use them EXACTLY as written in the Entity Context
8. If an example is marked [★ VERY SIMILAR], replicate its structure — only swap entity names/properties
9. If MATCH traverses nodes not included in RETURN, use RETURN DISTINCT to avoid duplicates
10. If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.
11. For queries with numeric filtering/ordering (highest, lowest, greatest, etc.), always exclude NULL values with WHERE clauses like `WHERE expenseRatio IS NOT NULL` or `WHERE expenseRatio > 0`.
12. ENTITY FILTERS: NEVER add ticker or name filters unless the entity is explicitly named in the question or present in Entity Context. For general/global questions ("which company...", "which entity...", "find companies...") without a specific name → omit all entity filters (ticker, name) and search globally across all nodes.
13. When asked about since inception or this year never filter by the year property just print the ordered results by the year

{entity_context}

Examples:
{examples}

Question: {question}

Cypher Query:"""

CYPHER_RETRY_TEMPLATE_LEAN = """You are a Neo4j Cypher expert. Fix the invalid Cypher query below.

Follow strictly the correct syntax from the examples:
{examples}

Failed query:
{failed_query}

Validation error:
{validation_errors}
{error_examples}

Rules:
1. Output ONLY the corrected Cypher — no explanations, no markdown
2. Fix ALL errors listed above
3. Do NOT use MERGE, CREATE, SET, DELETE

Corrected Cypher Query:"""

CYPHER_RETRY_TEMPLATE = """You are a Neo4j Cypher expert. Your previous Cypher query had errors. Fix them.

Neo4j Schema:
{schema}

{entity_context}

Examples:
{examples}

Original Question: {question}

Your Previous (Invalid) Query:
{failed_query}

Validation Errors:
{validation_errors}
{error_examples}

Rules:
1. Output ONLY the corrected Cypher query, no explanations or markdown
2. Fix ALL the validation errors listed above
3. Study the error examples provided - they show you exactly how to fix this type of error
4. Use ONLY node labels, relationship types, and property names that exist in the schema
5. Use proper Cypher syntax
6. Do NOT use MERGE, CREATE, SET, DELETE, or any write operations
7. FILING10K HAS NO YEAR PROPERTY: Never write `Filing10K {{year: X}}`. Year is on the REPORTS_IN relationship: use `WHERE r.year = X` instead.

Corrected Cypher Query:"""
