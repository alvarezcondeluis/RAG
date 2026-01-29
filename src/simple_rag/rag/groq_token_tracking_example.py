#!/usr/bin/env python3
"""
Example: Groq Token Tracking and Rate Limiting

Demonstrates how the CypherTranslator automatically tracks token usage
and enforces rate limits when using the Groq backend.
"""

from neo4j import GraphDatabase
from simple_rag.rag.text2cypher import CypherTranslator

# Initialize Neo4j driver
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Initialize with Groq backend
print("=" * 80)
print("🚀 Initializing CypherTranslator with Groq backend")
print("=" * 80)

translator = CypherTranslator(
    neo4j_driver=driver,
    model_name="llama-3.3-70b-versatile",
    backend="groq",
    temperature=0.2,
    use_entity_resolver=True
)

# Example queries
queries = [
    "What is the expense ratio for VTI?",
    "Show me the top 5 holdings for VGT",
    "What are the management fees for VHYAX?",
]

print("\n" + "=" * 80)
print("📝 Running Example Queries")
print("=" * 80)

for i, query in enumerate(queries, 1):
    print(f"\n--- Query {i}/{len(queries)} ---")
    print(f"Question: {query}")
    
    # Translate query (automatically checks rate limits and tracks tokens)
    cypher = translator.translate(query)
    
    if cypher:
        print(f"✓ Success!")
    else:
        print(f"❌ Failed (rate limit or error)")
    
    print("-" * 80)

# Check usage statistics
print("\n" + "=" * 80)
print("📊 Final Token Usage Statistics")
print("=" * 80)

stats = translator.get_groq_usage_stats()
print(f"Tokens Used: {stats['tokens_used']}/{stats['daily_limit']}")
print(f"Tokens Remaining: {stats['tokens_remaining']}")
print(f"Usage: {stats['usage_percent']}%")
print(f"Total Requests: {stats['requests_count']}")
print(f"Resets in: {stats['reset_in_hours']:.2f} hours")

# Optional: Reset counters manually (for testing)
# translator.reset_groq_usage()

driver.close()
print("\n✅ Example complete!")
