"""
Query Handler — orchestrates classification → schema selection → Text2Cypher / vector search.

This module is the single entry-point for answering a user query.
It uses the trained SetFit classifier to decide which *schema slice*
to feed to the Text2Cypher LLM, drastically reducing prompt size
and improving accuracy.

Usage:
    from simple_rag.rag.query_handler import QueryHandler

    handler = QueryHandler(neo4j_driver=driver)
    result  = handler.handle("What is the expense ratio of VTI?")
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from simple_rag.rag.query.query_classification import QueryClassifier, QueryCategory, LABELS
from simple_rag.rag.text2cypher import CypherTranslator
from simple_rag.rag.schema_definitions import DETAILED_SCHEMA, DETAILED_SCHEMA_V2
from simple_rag.rag.schema_slices import get_schema_for_category, get_merged_schema, DEFAULT_SCHEMA_VERSION
from simple_rag.rag.context_enrichment import ContextEnricher
from simple_rag.rag.post_processing.cypher_validator import ResultValidator

logger = logging.getLogger(__name__)


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Container for the full pipeline result."""
    query: str
    category: str
    confidence: float
    schema_used: str
    cypher: Optional[str] = None
    data: Any = None
    error: Optional[str] = None
    requires_vector_search: bool = False
    enrichment: Dict[str, Any] = field(default_factory=dict)
    query_embedding: Optional[List[float]] = None  # eager-embedded; passed as $queryVector at exec time


# ─── Handler ──────────────────────────────────────────────────────────────────

class QueryHandler:
    """
    Orchestrates: classify → pick schema → translate → (optionally execute).

    Args:
        neo4j_driver:     Active Neo4j driver instance.
        classifier_path:  Path to the trained SetFit model directory.
                          Defaults to ``query/models/query_classifier``
                          relative to this file.
        cypher_backend:   Backend for CypherTranslator ('ollama', 'groq', 'huggingface').
        cypher_model:     Model name for the CypherTranslator.
        confidence_threshold: Minimum confidence to trust the classifier.
                              Below this the handler falls back to ``cross_entity``
                              (the widest schema).
        **cypher_kwargs:  Extra keyword arguments forwarded to CypherTranslator.
    """

    def __init__(
        self,
        neo4j_driver,
        classifier_path: Optional[str] = None,
        cypher_backend: str = "ollama",
        cypher_model: str = "llama3.1:8b",
        confidence_threshold: float = 0.45,
        embedder=None,
        embedder_model: str = "nomic-ai/nomic-embed-text-v1.5",
        enable_query_embedding: bool = True,
        default_k: int = 5,
        few_shot_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        **cypher_kwargs,
    ):
        # ── Classifier ──────────────────────────────────────────────
        if classifier_path is None:
            classifier_path = str(
                Path(__file__).parent / "query" / "models" / "query_classifier"
            )
        self.classifier = QueryClassifier(classifier_path)

        # ── Text2Cypher translator ──────────────────────────────────
        self.translator = CypherTranslator(
            neo4j_driver=neo4j_driver,
            model_name=cypher_model,
            backend=cypher_backend,
            few_shot_embedding_model=few_shot_embedding_model,
            prompt_template_version=schema_version,
            **cypher_kwargs,
        )

        self.neo4j_driver = neo4j_driver
        self.confidence_threshold = confidence_threshold
        self.schema_version = schema_version

        # Patch translator's full schema so retries also use the right version
        if schema_version == "v2":
            self.translator.detailed_schema = DETAILED_SCHEMA_V2
            self.translator.schema = DETAILED_SCHEMA_V2

        # ── Query embedder (eager — every query is embedded so vector search is always available) ──
        self.enable_query_embedding = enable_query_embedding
        self.default_k = default_k
        self._embedder = None
        self._embedder_model = embedder_model
        # Only reuse the pre-computed query vector for few-shot if both models are identical.
        # If they differ, dimensions/semantics won't match the FAISS index.

        self._embedding_cache: Dict[str, List[float]] = {}
        if enable_query_embedding:
            if embedder is not None:
                self._embedder = embedder
                print("✓ Query embedder injected")
            else:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                self._embedder = HuggingFaceEmbedding(
                    model_name=embedder_model,
                    trust_remote_code=True,
                    cache_folder="./cache",
                )
                print(f"✓ Query embedder loaded: {embedder_model}")

        # Context enrichment engine (reuses the translator's entity resolver)
        self.enricher = ContextEnricher(
            neo4j_driver=neo4j_driver,
            entity_resolver=self.translator.entity_resolver,
        )

        print("✓ QueryHandler ready")

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """Embed a user query for vector search. Cached on the query string."""
        if not self.enable_query_embedding or self._embedder is None:
            return None
        if query in self._embedding_cache:
            return self._embedding_cache[query]
        try:
            embedding = self._embedder.get_query_embedding(query)
            self._embedding_cache[query] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return None

    # ── public API ────────────────────────────────────────────────────────

    def handle(
        self,
        user_query: str,
        execute: bool = False,
        temperature: Optional[float] = None,
        use_schema_injection: bool = True,
    ) -> QueryResult:
        """
        Full pipeline: classify → select schema → translate → optionally execute.

        Args:
            user_query:           Natural-language question.
            execute:              If True, run the generated Cypher against Neo4j
                                  and attach the rows to ``result.data``.
            temperature:          Override temperature for the Text2Cypher LLM.
            use_schema_injection: If True (default), the query classifier selects a
                                  focused schema slice for the initial LLM call.
                                  If False, the full DETAILED_SCHEMA is used from
                                  the start (classification still runs for routing,
                                  but the schema sent to the LLM is always complete).
                                  Note: retries always use the full schema regardless.

        Returns:
            QueryResult with category, cypher, and optional data.
        """
        import time
        start_pipeline = time.time()

        # Step 0 — embed the query.
        # Nomic ($queryVector) is only needed when execute=True — skip it during benchmarking.
        # MiniLM (few-shot FAISS) is embedded later inside translate(), only if needed.
        query_embedding = None
        if execute and self.enable_query_embedding:
            start_embedding = time.time()
            query_embedding = self._embed_query(user_query)
            embedding_time = time.time() - start_embedding
            if query_embedding is not None:
                print(f"🧬 Query embedded ({len(query_embedding)}d) in {embedding_time*1000:.0f}ms")

        # Step 1 — classify
        start_classification = time.time()
        prediction = self.classifier.predict(user_query)
        classification_time = time.time() - start_classification

        active_labels: list = prediction["labels"]          # e.g. ["fund_basic", "fund_portfolio"]
        top_label: str       = prediction["top_label"]       # highest-confidence label
        confidence: float    = prediction["confidence"]
        per_conf: dict       = prediction["per_label_confidence"]

        # Display
        label_str = " | ".join(
            f"{l} ({per_conf[l]:.2%})" for l in active_labels
        )
        logger.info(f"Classification: {label_str}")
        print(f"🏷️  Active labels: {label_str}")

        # Step 2 — pick schema slice(s)
        if QueryCategory.NOT_RELATED.value in active_labels:
            result = QueryResult(
                query=user_query, category=QueryCategory.NOT_RELATED.value,
                confidence=confidence, schema_used="",
            )
            result.error = (
                "NOT_RELATED: This question is outside the scope of the SEC Filings Intelligence system. "
                "I can only answer questions about fund holdings, expense ratios, returns, investment strategies, "
                "company 10-K filings, financial metrics, executive compensation, and insider transactions. "
                "Please rephrase your question to relate to SEC filings or fund data."
            )
            print("🚫 Not related — skipping.")
            return result

        if len(active_labels) == 1:
            selected_schema_name = active_labels[0]
            schema_slice = get_schema_for_category(active_labels[0], version=self.schema_version)
            print(f"→ Single label: {active_labels[0]} ({confidence:.2%})")
        else:
            selected_schema_name = " + ".join(active_labels)
            schema_slice = get_merged_schema(active_labels, version=self.schema_version)
            print(f"🔀 Multi-label: [{', '.join(active_labels)}] — merging schemas")

        # Step 3 — build result container
        result = QueryResult(
            query=user_query,
            category=top_label,
            confidence=confidence,
            schema_used=selected_schema_name,
            query_embedding=query_embedding,
        )

        # fund_profile → also flag vector search
        if QueryCategory.FUND_PROFILE.value in active_labels:
            result.requires_vector_search = True
            print("🔀 Hybrid — will generate Cypher AND flag vector search.")

        # Step 4 — translate with the sliced schema (or full schema if injection disabled)
        if use_schema_injection:
            effective_schema = schema_slice
            print(f"📐 Schema: slice ({selected_schema_name})")
        else:
            effective_schema = DETAILED_SCHEMA_V2 if self.schema_version == "v2" else DETAILED_SCHEMA
            
        start_translation = time.time()
        cypher = self._translate_with_schema(
            user_query, effective_schema, temperature=temperature,
        )
        translation_time = time.time() - start_translation
        result.cypher = cypher

        if cypher is None:
            blocked_op = getattr(self.translator, "last_write_blocked_op", None)
            if blocked_op:
                result.error = f"WRITE_BLOCKED:{blocked_op}"
            else:
                result.error = "Failed to generate Cypher query."
            return result

        # Step 5 — optionally execute (always pass $queryVector + $k; Neo4j ignores unused params)
        if execute and cypher:
            params = {"k": self.default_k}
            if query_embedding is not None:
                params["queryVector"] = query_embedding
            result.data = self._execute(cypher, parameters=params)

        # Step 5a — result-level semantic validation with one retry
        # Catches data-quality issues (e.g. all expenseRatio = 0.0) that only
        # become visible after execution, outside the syntax/schema retry loop.
        if execute and result.data is not None:
            result_issue = ResultValidator.validate(cypher, result.data)
            if result_issue is not None:
                print(f"⚠️  Result validation [{result_issue.triggered_rule}] — retrying with constraint...")
                error_hint = result_issue.all_errors[0]
                augmented_query = f"{user_query}\n[DATA QUALITY CONSTRAINT: {error_hint}]"
                retry_cypher = self._translate_with_schema(
                    augmented_query, effective_schema, temperature=temperature,
                        )
                if retry_cypher:
                    result.cypher = retry_cypher
                    result.data = self._execute(retry_cypher, parameters=params)
                    print("✅ Result-validated retry complete")

        # Step 6 — context enrichment (supplementary queries for richer LLM context)
        if execute and result.data is not None:
            result.enrichment = self.enricher.enrich(
                category=top_label,
                user_query=user_query,
                main_results=result.data,
            )

        total_pipeline_time = time.time() - start_pipeline
        print(f"\n⏱️  Pipeline timings:")
        print(f"   - Classification (CPU): {classification_time:.3f}s")
        print(f"   - Translation (LLM):    {translation_time:.3f}s")
        print(f"   - Total Pipeline Time:  {total_pipeline_time:.3f}s")

        return result

    # ── internals ─────────────────────────────────────────────────────────

    def _translate_with_schema(
        self,
        user_query: str,
        schema_slice: str,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """
        Temporarily swap the translator's schema, call translate(),
        then restore the original schema.
        """
        original_schema = self.translator.schema
        try:
            self.translator.schema = schema_slice
            return self.translator.translate(user_query, temperature=temperature)
        finally:
            self.translator.schema = original_schema

    def _execute(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Run a Cypher query (with optional parameters) and return the result rows."""
        params = parameters or {}
        try:
            with self.neo4j_driver.session() as session:
                records = session.run(cypher, **params)
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            print(f"❌ Execution error: {e}")
            return None
