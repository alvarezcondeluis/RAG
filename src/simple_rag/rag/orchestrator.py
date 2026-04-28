"""
Terminal orchestrator for the SEC Filings Intelligence pipeline.

Provides interactive configuration of:
- Text2Cypher backend and model selection
- Answer generation LLM provider and model selection
- Pipeline options (schema injection, entity resolution, few-shot, verbose)

Then runs a REPL loop for querying the knowledge graph.

Usage:
    python -m simple_rag.rag.orchestrator
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure project root on path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

load_dotenv(_project_root / ".env")


# ── Pipeline configuration ───────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Holds all user-selected pipeline options."""
    # Text2Cypher
    cypher_backend: str = "ollama"
    cypher_model: str = "llama3.1:8b"

    # Answer generation
    answer_provider_name: str = "groq"
    answer_model: str = "llama-3.3-70b-versatile"

    # Pipeline toggles
    use_schema_injection: bool = True
    enable_entity_resolution: bool = True
    enable_few_shot: bool = True
    verbose: bool = False


# ── Interactive setup ────────────────────────────────────────────────────────

TEXT2CYPHER_BACKENDS = [
    ("ollama", "Ollama (local)", "tomasonjo/llama3-text2cypher-demo:8b_4bit"),
    ("groq", "Groq", "llama-3.3-70b-versatile"),
    ("openai", "OpenAI-compatible (llama.cpp / LM Studio)", ""),
]


def _ask_choice(prompt: str, options: list[str], default: int = 1) -> int:
    """Ask user to pick from numbered options. Returns 0-based index."""
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "" and default is not None:
                return default - 1
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"  Please enter a number between 1 and {len(options)}")
        except (ValueError, EOFError):
            print(f"  Please enter a number between 1 and {len(options)}")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question with a default."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            raw = input(f"{prompt} {suffix}: ").strip().lower()
            if raw == "":
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            print("  Please enter y or n")
        except EOFError:
            return default


def setup_pipeline() -> PipelineConfig:
    """Interactive terminal setup — returns a PipelineConfig."""
    from simple_rag.rag.llm_providers.registry import ProviderRegistry

    config = PipelineConfig()
    registry = ProviderRegistry()

    print("\n╔══════════════════════════════════════════════╗")
    print("║  SEC Filings Intelligence — Pipeline Setup   ║")
    print("╚══════════════════════════════════════════════╝")

    # ── 1. Text2Cypher backend ───────────────────────────────────
    print("\n─── Text2Cypher Configuration ───")
    for i, (key, display, default_model) in enumerate(TEXT2CYPHER_BACKENDS, 1):
        extra = f" (default: {default_model})" if default_model else ""
        print(f"  [{i}] {display}{extra}")

    idx = _ask_choice("> Select text2cypher backend: ", [b[1] for b in TEXT2CYPHER_BACKENDS])
    backend_key, backend_display, default_model = TEXT2CYPHER_BACKENDS[idx]
    config.cypher_backend = backend_key

    # Model selection for text2cypher
    if backend_key == "groq":
        # Use registry to list Groq models
        try:
            groq_prov = registry.get_provider("groq")
            models = groq_prov.list_models()
            if models:
                print(f"\n  Available models on Groq:")
                display_models = models[:15]
                for i, m in enumerate(display_models, 1):
                    ctx = f" ({m.context_window:,} ctx)" if m.context_window else ""
                    print(f"  [{i}] {m.id}{ctx}")
                idx = _ask_choice("> Select text2cypher model: ", [m.id for m in display_models])
                config.cypher_model = display_models[idx].id
            else:
                config.cypher_model = default_model
        except Exception:
            config.cypher_model = default_model
    elif backend_key == "ollama":
        # List local Ollama models
        try:
            ollama_prov = registry.get_provider("ollama")
            models = ollama_prov.list_models()
            if models:
                print(f"\n  Available models on Ollama:")
                for i, m in enumerate(models, 1):
                    desc = f" — {m.description}" if m.description else ""
                    print(f"  [{i}] {m.id}{desc}")
                idx = _ask_choice("> Select text2cypher model: ", [m.id for m in models])
                config.cypher_model = models[idx].id
            else:
                print(f"  No models found. Using default: {default_model}")
                config.cypher_model = default_model
        except Exception:
            config.cypher_model = default_model
    elif backend_key == "openai":
        model_name = input("> Enter model name (or press Enter for default): ").strip()
        config.cypher_model = model_name if model_name else "qwen2.5-coder"

    print(f"  ✓ Text2Cypher: {backend_display} / {config.cypher_model}")

    # ── 2. Answer generation LLM ─────────────────────────────────
    answer_provider, answer_model = registry.interactive_select(
        prompt_label="Answer Generation LLM"
    )
    config.answer_provider_name = answer_provider.provider_name.lower()
    config.answer_model = answer_model

    # ── 3. Pipeline options ──────────────────────────────────────
    print("\n─── Pipeline Options ───")
    config.use_schema_injection = _ask_yn("  Use schema injection? (recommended)")
    config.enable_entity_resolution = _ask_yn("  Enable entity resolution?")
    config.enable_few_shot = _ask_yn("  Enable few-shot examples?")
    config.verbose = _ask_yn("  Verbose mode (show Cypher + timings)?", default=False)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'─' * 48}")
    print(f"  Text2Cypher:       {config.cypher_backend} / {config.cypher_model}")
    print(f"  Answer LLM:        {config.answer_provider_name} / {config.answer_model}")
    print(f"  Schema injection:  {'ON' if config.use_schema_injection else 'OFF'}")
    print(f"  Entity resolution: {'ON' if config.enable_entity_resolution else 'OFF'}")
    print(f"  Few-shot examples: {'ON' if config.enable_few_shot else 'OFF'}")
    print(f"  Verbose:           {'ON' if config.verbose else 'OFF'}")
    print(f"{'─' * 48}")

    return config


# ── Pipeline execution ───────────────────────────────────────────────────────

def _init_pipeline(config: PipelineConfig):
    """Initialize QueryHandler and answer LLM from config."""
    import neo4j as neo4j_lib
    from simple_rag.database.neo4j.config import settings
    from simple_rag.rag.query_handler import QueryHandler
    from simple_rag.rag.llm_providers.registry import ProviderRegistry
    from simple_rag.rag.answer_generation.prompt_templates import (
        ANSWER_SYSTEM_PROMPT,
        build_answer_prompt,
    )
    from simple_rag.rag.answer_generation.result_classifier import ResultClassifier

    # Neo4j driver
    print("\n  Connecting to Neo4j...")
    driver = neo4j_lib.GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("  ✓ Neo4j connected")

    # QueryHandler with text2cypher config
    cypher_kwargs = {}
    if not config.enable_entity_resolution:
        cypher_kwargs["use_entity_resolution"] = False
    if not config.enable_few_shot:
        cypher_kwargs["use_few_shot"] = False

    handler = QueryHandler(
        neo4j_driver=driver,
        cypher_backend=config.cypher_backend,
        cypher_model=config.cypher_model,
        **cypher_kwargs,
    )

    # Answer LLM provider
    registry = ProviderRegistry()
    answer_provider = registry.get_provider(
        config.answer_provider_name, model_id=config.answer_model
    )

    return handler, answer_provider, driver


def run_loop(config: PipelineConfig):
    """Main REPL loop — ask questions, get answers."""
    from simple_rag.rag.answer_generation.prompt_templates import (
        ANSWER_SYSTEM_PROMPT,
        build_answer_prompt,
    )
    from simple_rag.rag.answer_generation.result_classifier import ResultClassifier, ResultType
    from simple_rag.rag.context_enrichment import format_enrichment_context, resolve_document_provenance

    handler, answer_provider, driver = _init_pipeline(config)
    classifier = ResultClassifier()

    print("\n✓ Pipeline ready. Type your question or 'quit' to exit.")
    print("─" * 48)

    try:
        while True:
            try:
                query = input("\n\033[1m> \033[0m").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            try:
                # Step 1: Text2Cypher + Execute
                start = time.time()
                result = handler.handle(
                    query,
                    execute=True,
                    use_schema_injection=config.use_schema_injection,
                )
                cypher_time = time.time() - start

                if config.verbose:
                    print(f"\n  \033[90mCategory: {result.category} ({result.confidence:.2%})\033[0m")
                    if result.cypher:
                        print(f"  \033[90mCypher: {result.cypher}\033[0m")
                    print(f"  \033[90mCypher pipeline: {cypher_time:.2f}s\033[0m")

                if result.error:
                    print(f"\n  \033[91m⚠ {result.error}\033[0m")
                    continue

                if result.data is None:
                    print("\n  \033[93mNo results returned from the database.\033[0m")
                    continue

                if config.verbose:
                    print(f"  \033[90mResults: {len(result.data)} rows\033[0m")

                # Step 2: Classify result type
                result_type = classifier.classify(result.data, result.category)

                # Step 3: Build prompt and generate answer
                enrichment_text = format_enrichment_context(result.enrichment)
                provenance_text = resolve_document_provenance(
                    cypher=result.cypher or "",
                    neo4j_driver=driver,
                    main_results=result.data,
                )
                user_prompt = build_answer_prompt(
                    user_query=query,
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

                print()
                # Stream the answer
                start = time.time()
                for token in answer_provider.stream(messages, temperature=0.1):
                    print(token, end="", flush=True)
                answer_time = time.time() - start
                print()  # newline after streaming

                if config.verbose:
                    print(f"\n  \033[90mAnswer generation: {answer_time:.2f}s\033[0m")
                    print(f"  \033[90mResult type: {result_type.value}\033[0m")
                    print(f"  \033[90mTotal: {cypher_time + answer_time:.2f}s\033[0m")

                # Step 4: Show charts/tables if present
                if result_type == ResultType.CHART_SVG:
                    charts = classifier.extract_svg_data(result.data)
                    if charts:
                        print(f"\n  \033[92m📊 {len(charts)} chart(s) available (render in Streamlit)\033[0m")

                if result_type == ResultType.HOLDINGS_TABLE and len(result.data) > 0:
                    tabular = classifier.extract_tabular_data(result.data)
                    if tabular and len(tabular) <= 20:
                        _print_table(tabular)

            except KeyboardInterrupt:
                print("\n  (interrupted)")
                continue
            except Exception as e:
                print(f"\n  \033[91mError: {e}\033[0m")
                if config.verbose:
                    import traceback
                    traceback.print_exc()

    finally:
        driver.close()
        print("\n  Goodbye.")


def _print_table(rows: list[dict]):
    """Simple terminal table for small result sets."""
    if not rows:
        return
    keys = list(rows[0].keys())
    # Truncate wide values
    col_widths = {k: max(len(k), max(len(str(r.get(k, ""))[:30]) for r in rows)) for k in keys}

    header = " | ".join(k.ljust(col_widths[k]) for k in keys)
    sep = "-+-".join("-" * col_widths[k] for k in keys)
    print(f"\n  {header}")
    print(f"  {sep}")
    for row in rows:
        line = " | ".join(str(row.get(k, ""))[:30].ljust(col_widths[k]) for k in keys)
        print(f"  {line}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    config = setup_pipeline()
    run_loop(config)


if __name__ == "__main__":
    main()
