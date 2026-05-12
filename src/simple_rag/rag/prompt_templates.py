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
{error_examples}

Rules:
1. Output ONLY the corrected Cypher query, no explanations or markdown
2. Fix ALL the validation errors listed above
3. Study the error examples provided - they show you exactly how to fix this type of error
4. Use ONLY node labels, relationship types, and property names that exist in the schema
5. Use proper Cypher syntax
6. Do NOT use MERGE, CREATE, SET, DELETE, or any write operations

Corrected Cypher Query:"""
