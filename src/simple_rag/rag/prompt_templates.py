"""
Prompt Templates for Text2Cypher Translation.

This module contains the LLM prompt templates used for generating and
correcting Cypher queries from natural language.
"""

CYPHER_GENERATION_TEMPLATE = """You are a Neo4j Cypher expert. Convert the question below to a valid Cypher query using the schema strictly.

Schema:
{schema}

Examples:
{examples}

{entity_context}

CRITICAL RULES:
1. OUTPUT FORMAT: Output ONLY the raw Cypher query. No explanations, no markdown formatting, no ```cypher blocks.
2. UNIQUE VARIABLES (FATAL ERROR PREVENTION): NEVER use the same variable name for both a relationship and a node. For example, `-[p:HAS_PORTFOLIO]->(p:Portfolio)` is invalid and will crash. Always use distinct names like `-[rel:HAS_PORTFOLIO]->(p:Portfolio)`.
3. SUBQUERY SYNTAX: Do NOT use SQL-style subqueries like `ticker IN (MATCH...)`. To filter based on a sub-pattern, use `WHERE EXISTS {{ MATCH ... }}` or use a `WITH` clause.
4. AGGREGATIONS & GROUPING: If the question asks for a global calculation (e.g., highest, average, total across all nodes), DO NOT include row-specific identifiers (like ticker, name, or year) in the RETURN clause. Doing so triggers an implicit GROUP BY and ruins the calculation.
5. EXTRA PROPERTIES: For standard lists/queries (NOT aggregations), always return extra identifying properties (ticker, name) and relationship properties beyond what was asked.
6. SCHEMA STRICTNESS: Use property names EXACTLY as they appear in the provided schema. Do not hallucinate properties like `.text` if the schema specifies `.summaryProspectus`.
7. FILTERING: Take into account the score of the entity resolver and use it to filter the results by ticker or name.
8. DUPLICATES: If a MATCH traverses nodes not included in the RETURN, use RETURN DISTINCT.
9. FEW-SHOT REPLICATION: If an example is marked [★ VERY SIMILAR], strictly replicate its structural logic — only swap the specific entity names or target properties.
10. FULLTEXT INDEXES: If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.
11. AVERAGE RETURNS: Use predefined properties like return1y, return5y, return10y instead of calculating based on dates.
12. LATEST: When asked for the latest ALWAYS use ORDER BY property.date DESC LIMIT 1 rather than building complex NOT EXISTS subqueries
13. TICKER: Use the exact ticker from the question.
14. NO YEAR FILTER UNLESS ASKED: Only add year/date filters if the question explicitly mentions a specific year or date range.
15. NO TEXT FILTERS ON VECTOR SEARCH: Never add WHERE CONTAINS or title filters on Section/Chunk nodes in vector searches. Embeddings already rank by relevance. Text filters silently return 0 results (section titles like 'Risk Factors' won't match topic phrases).
16. GENERAL QUERIES: Questions asking broadly ("which funds...", "find funds...") without naming a specific entity → search globally, omit entity filters.
17. ENTITY FILTERS IN VECTOR SEARCH: Add entity filter after YIELD only if BOTH: (a) Entity Context score ≈ 100 (exact match) AND (b) ticker/entity explicitly stated in question text. Never use CONTAINS on entity properties.

Question: {question}

Cypher Query:"""

CYPHER_GENERATION_TEMPLATE2 = """You are a Neo4j Cypher expert. Convert the question below to a valid Cypher query using the schema strictly.

Schema:
{schema}

Examples:
{examples}

{entity_context}

Rules:
1. Output ONLY the Cypher query — no explanations, no markdown
2. Use MATCH, WHERE (only after MATCH, never after RETURN), RETURN, COUNT(*) with correct syntax
3. Use property names exactly as in the schema
4. For text search use CONTAINS or regex; for numeric comparisons use >, <, =, etc.
5. Always return extra identifying properties (ticker, name, etc.) beyond what was asked
6. Always return relationship properties
7. If entity names appear in Entity Context, use them EXACTLY as written
8. If an example is marked [★ VERY SIMILAR], replicate its structure — only swap entity names/properties
9. If MATCH traverses nodes not included in RETURN, use RETURN DISTINCT to avoid duplicates
10. If the entity extractor has identified the entity with a score similar to 100 do not use the index, use a ticker or name search.

Question: {question}

Cypher Query:"""

CYPHER_RETRY_TEMPLATE_LEAN = """You are a Neo4j Cypher expert. Fix the invalid Cypher query below.

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

Corrected Cypher Query:"""
