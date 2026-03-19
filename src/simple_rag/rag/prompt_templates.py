"""
Prompt Templates for Text2Cypher Translation.

This module contains the LLM prompt templates used for generating and
correcting Cypher queries from natural language.
"""

CYPHER_GENERATION_TEMPLATE = """You are a Neo4j Cypher expert. Convert the natural language question to a valid Cypher query.

Neo4j Schema:
{schema}

Examples:
{examples}

{entity_context}

Rules:
1. Output ONLY the Cypher query, no explanations or markdown
2. Use proper Cypher syntax with MATCH, WHERE, RETURN, COUNT(*)
3. Use property names exactly as shown in schema
4. For numeric comparisons, use appropriate operators (>, <, =, etc.) just after the MATCH clause
5. For text search, use CONTAINS or regular expressions
6. Always return more properties than the ones asked. Include ticker, name, etc.
7. If entity names are provided in the Entity Context, use them EXACTLY as shown
8. Never use a WHERE statement or any filtering condition immediately after a RETURN statement.
9. Always return the relationships properties
10. If an example is marked [★ VERY SIMILAR], replicate its exact query structure and clause order — only adapt the entity names or properties for the current question.
11. When MATCH traverses relationships to intermediate nodes NOT included in RETURN, always write RETURN DISTINCT to avoid duplicate rows (e.g., if you MATCH (f:Fund)->(:Portfolio)->(:Holding) but only RETURN f.name, write RETURN DISTINCT f.name).

Question: {question}

Cypher Query:"""

CYPHER_RETRY_TEMPLATE = """You are a Neo4j Cypher expert. Your previous Cypher query had errors. Fix them.

Neo4j Schema:
{schema}

{entity_context}

Original Question: {question}

Your Previous (Invalid) Query:
{failed_query}

Validation Errors:
{validation_errors}

Rules:
1. Output ONLY the corrected Cypher query, no explanations or markdown
2. Fix ALL the validation errors listed above
3. Use ONLY node labels, relationship types, and property names that exist in the schema
4. Use proper Cypher syntax
5. Do NOT use MERGE, CREATE, SET, DELETE, or any write operations

Corrected Cypher Query:"""
