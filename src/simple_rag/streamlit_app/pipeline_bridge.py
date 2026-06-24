"""
Pipeline bridge — wraps the RAG orchestrator for Streamlit consumption.

Provides non-interactive initialization and query processing that Streamlit
can drive through session_state.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import neo4j as neo4j_lib

from simple_rag.database.neo4j.config import settings
from simple_rag.rag.orchestrator import PipelineConfig, TEXT2CYPHER_BACKENDS
from simple_rag.rag.llm_providers.base import LLMProvider, ModelInfo
from simple_rag.rag.llm_providers.registry import ProviderRegistry
from simple_rag.rag.answer_generation.result_classifier import ResultClassifier, ResultType

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Holds initialized pipeline objects for persistence in session_state."""
    config: PipelineConfig
    handler: Any  # QueryHandler
    answer_provider: LLMProvider
    driver: neo4j_lib.Driver
    classifier: ResultClassifier = field(default_factory=ResultClassifier)


@dataclass
class QueryStepUpdate:
    """Progress update emitted during query processing."""
    step: str
    detail: str
    elapsed: float = 0.0


@dataclass
class PipelineQueryResult:
    """Complete result from pipeline query processing."""
    category: str
    confidence: float
    cypher: Optional[str]
    data: Any
    result_type: ResultType
    enrichment_text: str
    provenance_text: str
    token_stream: Iterator[str]
    charts: list = field(default_factory=list)
    tabular: list = field(default_factory=list)
    error: Optional[str] = None
    answer_messages: Optional[list] = None


# ── Registry singleton ───────────────────────────────────────────────────────

_registry: Optional[ProviderRegistry] = None


def _get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


# ── Provider / model helpers ─────────────────────────────────────────────────

def get_available_providers() -> list[dict]:
    """Return providers with API key status."""
    return _get_registry().available_providers()


def list_models_for_provider(provider_name: str) -> list[ModelInfo]:
    """Fetch models from a provider. Returns empty list on failure."""
    try:
        provider = _get_registry().get_provider(provider_name)
        return provider.list_models()
    except Exception as e:
        logger.warning("Failed to list models for %s: %s", provider_name, e)
        return []


def get_text2cypher_backends() -> list[tuple[str, str, str]]:
    """Return the TEXT2CYPHER_BACKENDS list from the orchestrator."""
    return list(TEXT2CYPHER_BACKENDS)


def list_openai_compatible_models(host: str = "localhost", port: int = 1234) -> list[str]:
    """Fetch model IDs from an OpenAI-compatible server (e.g. LM Studio)."""
    from simple_rag.rag.orchestrator import fetch_openai_compatible_models
    return fetch_openai_compatible_models(host, port)


# ── Pipeline lifecycle ───────────────────────────────────────────────────────

def init_pipeline(config: PipelineConfig) -> PipelineState:
    """Initialize the full RAG pipeline from config.

    Raises on failure (Neo4j unreachable, invalid provider, etc.).
    """
    from simple_rag.rag.query_handler import QueryHandler

    # Neo4j driver
    driver = neo4j_lib.GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    driver.verify_connectivity()

    # QueryHandler
    cypher_kwargs: dict = {
        "use_entity_resolver": config.enable_entity_resolution,
        "use_few_shot": config.enable_few_shot,
        "retry_strategy": config.retry_strategy,
        "few_shot_embedding_model": config.few_shot_embedding_model,
    }
    if not config.retry_module:
        cypher_kwargs["max_validation_retries"] = 1
    if config.cypher_backend == "openai":
        cypher_kwargs["openai_compatible_host"] = config.openai_compatible_host
        cypher_kwargs["openai_compatible_port"] = config.openai_compatible_port
    if config.main_llm_model_openai:
        cypher_kwargs["main_llm_model_openai"] = config.main_llm_model_openai

    handler = QueryHandler(
        neo4j_driver=driver,
        cypher_backend=config.cypher_backend,
        cypher_model=config.cypher_model,
        enable_query_embedding=config.embed_vector_queries,
        **cypher_kwargs,
    )

    # Answer LLM provider
    registry = _get_registry()
    if config.answer_provider_name == "openai_local":
        from simple_rag.rag.llm_providers.openrouter_provider import OpenRouterProvider
        answer_provider = OpenRouterProvider(
            base_url=f"http://{config.openai_compatible_host}:{config.openai_compatible_port}/v1",
            api_key="local",
            model_id=config.answer_model,
        )
    else:
        answer_provider = registry.get_provider(
            config.answer_provider_name, model_id=config.answer_model
        )

    return PipelineState(
        config=config,
        handler=handler,
        answer_provider=answer_provider,
        driver=driver,
    )


def shutdown_pipeline(pipeline: PipelineState) -> None:
    """Close the Neo4j driver and clean up."""
    try:
        pipeline.driver.close()
    except Exception:
        pass


def verify_connection(pipeline: PipelineState) -> bool:
    """Check if Neo4j is still reachable."""
    try:
        pipeline.driver.verify_connectivity()
        return True
    except Exception:
        return False


# ── Query processing ─────────────────────────────────────────────────────────

def process_query(
    query: str,
    pipeline: PipelineState,
) -> tuple[list[QueryStepUpdate], PipelineQueryResult]:
    """Run the full RAG pipeline for a user query.

    Returns:
        (steps, result) where steps is a list of progress updates and
        result contains the data + token stream for the answer.
    """
    from simple_rag.rag.answer_generation.prompt_templates import (
        ANSWER_SYSTEM_PROMPT,
        build_answer_prompt,
    )
    from simple_rag.rag.answer_generation.result_enhancer import enhance
    from simple_rag.rag.context_enrichment import (
        format_enrichment_context,
        resolve_document_provenance,
    )

    steps: list[QueryStepUpdate] = []
    config = pipeline.config

    # Step 1: Text2Cypher + Execute
    t0 = time.time()
    result = pipeline.handler.handle(
        query,
        execute=True,
        use_schema_injection=config.use_schema_injection,
    )
    cypher_time = time.time() - t0

    steps.append(QueryStepUpdate(
        step="cypher",
        detail=f"Category: {result.category} ({result.confidence:.0%})",
        elapsed=cypher_time,
    ))

    if result.error:
        return steps, PipelineQueryResult(
            category=result.category,
            confidence=result.confidence,
            cypher=result.cypher,
            data=None,
            result_type=ResultType.EMPTY,
            enrichment_text="",
            provenance_text="",
            token_stream=iter([]),
            error=result.error,
        )

    if result.data is None:
        return steps, PipelineQueryResult(
            category=result.category,
            confidence=result.confidence,
            cypher=result.cypher,
            data=None,
            result_type=ResultType.EMPTY,
            enrichment_text="",
            provenance_text="",
            token_stream=iter([]),
            error="No results returned from the database.",
        )

    # Step 2: Enhance results (empty check + row truncation)
    enhanced = enhance(result.data, query=query)
    if enhanced.is_empty:
        return steps, PipelineQueryResult(
            category=result.category,
            confidence=result.confidence,
            cypher=result.cypher,
            data=[],
            result_type=ResultType.EMPTY,
            enrichment_text="",
            provenance_text="",
            token_stream=iter([]),
            error=enhanced.empty_message,
        )

    # Step 3: Classify result type
    result_type = pipeline.classifier.classify(enhanced.records, result.category)

    # Step 4: Enrichment + provenance
    t1 = time.time()
    enrichment_text = format_enrichment_context(result.enrichment)
    provenance_text = resolve_document_provenance(
        cypher=result.cypher or "",
        neo4j_driver=pipeline.driver,
        main_results=enhanced.records,
    )
    enrich_time = time.time() - t1

    record_detail = f"{enhanced.original_count} records"
    if enhanced.truncated:
        record_detail += f" (showing {len(enhanced.records)})"
    record_detail += f", type: {result_type.value}"

    steps.append(QueryStepUpdate(
        step="enrich",
        detail=record_detail,
        elapsed=enrich_time,
    ))

    # Step 5: Build prompt
    user_prompt = build_answer_prompt(
        user_query=query,
        neo4j_results=enhanced.records,
        result_type=result_type,
        query_category=result.category,
        enrichment_context=enrichment_text,
        provenance_context=provenance_text,
        truncation_note=enhanced.truncation_note,
    )

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Create token stream (lazy — streamed by Streamlit)
    token_stream = pipeline.answer_provider.stream(messages, temperature=0.1)

    # Extract charts / tabular data
    charts = []
    tabular = []
    if result_type == ResultType.CHART_SVG:
        charts = pipeline.classifier.extract_svg_data(enhanced.records) or []
    if result_type == ResultType.HOLDINGS_TABLE and enhanced.records:
        tabular = pipeline.classifier.extract_tabular_data(enhanced.records) or []

    return steps, PipelineQueryResult(
        category=result.category,
        confidence=result.confidence,
        cypher=result.cypher,
        data=enhanced.records,
        result_type=result_type,
        enrichment_text=enrichment_text,
        provenance_text=provenance_text,
        token_stream=token_stream,
        charts=charts,
        tabular=tabular,
        answer_messages=messages,
    )
