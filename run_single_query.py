"""
Run a single question through the full RAG pipeline using Groq for both
Text2Cypher and Answer Generation.

Usage:
    uv run python run_single_query.py
    uv run python run_single_query.py "Your custom question here"
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root / "src"))
load_dotenv(_project_root / ".env")

QUESTION = "Show me the general information about the healthcare fund."

CYPHER_BACKEND = "groq"
CYPHER_MODEL = "llama-3.3-70b-versatile"
ANSWER_PROVIDER = "groq"
ANSWER_MODEL = "llama-3.3-70b-versatile"


def main():
    import neo4j as neo4j_lib
    from simple_rag.database.neo4j.config import settings
    from simple_rag.rag.query_handler import QueryHandler
    from simple_rag.rag.llm_providers.registry import ProviderRegistry
    from simple_rag.rag.answer_generation.prompt_templates import ANSWER_SYSTEM_PROMPT, build_answer_prompt
    from simple_rag.rag.answer_generation.result_classifier import ResultClassifier
    from simple_rag.rag.context_enrichment import format_enrichment_context, resolve_document_provenance

    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION

    print(f"\n{'═' * 60}")
    print(f"  Question: {question}")
    print(f"  Text2Cypher: {CYPHER_BACKEND} / {CYPHER_MODEL}")
    print(f"  Answer LLM:  {ANSWER_PROVIDER} / {ANSWER_MODEL}")
    print(f"{'═' * 60}\n")

    # Connect to Neo4j
    driver = neo4j_lib.GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("✓ Neo4j connected\n")

    # Initialize pipeline
    handler = QueryHandler(
        neo4j_driver=driver,
        cypher_backend=CYPHER_BACKEND,
        cypher_model=CYPHER_MODEL,
    )
    registry = ProviderRegistry()
    answer_provider = registry.get_provider(ANSWER_PROVIDER, model_id=ANSWER_MODEL)
    classifier = ResultClassifier()

    try:
        # Step 1: Classify, translate, and execute Cypher
        t0 = time.time()
        result = handler.handle(question, execute=True, use_schema_injection=True)
        cypher_time = time.time() - t0

        print(f"Category:    {result.category} ({result.confidence:.2%})")
        print(f"Cypher:      {result.cypher}")
        print(f"Pipeline:    {cypher_time:.2f}s")

        if result.error:
            print(f"\n⚠ Error: {result.error}")
            return

        if not result.data:
            print("\nNo results returned from the database.")
            return

        print(f"Rows:        {len(result.data)}\n")

        # Step 2: Classify result type and build answer prompt
        result_type = classifier.classify(result.data, result.category)
        enrichment_text = format_enrichment_context(result.enrichment)
        provenance_text = resolve_document_provenance(
            cypher=result.cypher or "",
            neo4j_driver=driver,
            main_results=result.data,
        )
        user_prompt = build_answer_prompt(
            user_query=question,
            neo4j_results=result.data,
            result_type=result_type,
            query_category=result.category,
            enrichment_context=enrichment_text,
            provenance_context=provenance_text,
        )

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Print the full prompt sent to the answer LLM
        print(f"\n{'━' * 60}")
        print("  SYSTEM PROMPT")
        print(f"{'━' * 60}")
        print(ANSWER_SYSTEM_PROMPT)
        print(f"\n{'━' * 60}")
        print("  USER PROMPT (with context)")
        print(f"{'━' * 60}")
        print(user_prompt)
        print(f"{'━' * 60}\n")

        # Step 3: Stream the answer
        print(f"{'─' * 60}")
        t1 = time.time()
        for token in answer_provider.stream(messages, temperature=0.1):
            print(token, end="", flush=True)
        answer_time = time.time() - t1
        print(f"\n{'─' * 60}")
        print(f"\nAnswer generation: {answer_time:.2f}s  |  Total: {cypher_time + answer_time:.2f}s")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
