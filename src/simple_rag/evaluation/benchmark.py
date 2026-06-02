import re
import sys
import builtins
import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ── Schema knowledge for static adherence checks ──────────────────────────────
_KNOWN_NODE_LABELS = {
    "Provider", "Trust", "Fund", "ShareClass", "Document", "Profile",
    "Section", "Objective", "PerformanceCommentary", "Risk", "RiskFactor", "Strategy",
    "Chunk", "Image", "Table", "Person", "AverageReturns", "Sector", "Region",
    "Portfolio", "Holding", "AssetCategory", "Company", "Filing10K",
    "Financials", "FinancialMetric", "Segment", "CompensationPackage",
    "InsiderTransaction", "FinancialHighlight", "BusinessInformation",
    "LegalProceeding", "ManagementDiscussion", "Properties",
}

_KNOWN_REL_TYPES = {
    "MANAGES", "ISSUES", "HAS_SHARE_CLASS", "EXTRACTED_FROM", "DEFINED_BY",
    "HAS_SECTION", "HAS_CHUNK", "HAS_CHART", "HAS_TABLE", "MANAGED_BY",
    "HAS_AVERAGE_RETURNS", "HAS_SECTOR_ALLOCATION", "HAS_REGION_ALLOCATION",
    "HAS_PORTFOLIO", "HAS_HOLDING", "OF_ASSET_TYPE", "REPRESENTS",
    "HAS_FINANCIAL_HIGHLIGHT", "REPORTS_IN", "HAS_FINANCIALS", "HAS_METRIC",
    "HAS_SEGMENT", "HAS_CEO", "RECEIVED_COMPENSATION", "AWARDED_BY",
    "DISCLOSED_IN", "HAS_INSIDER_TRANSACTION", "MADE_BY",
}


def _check_schema_adherence(cypher: str) -> tuple[bool, list[str]]:
    """Static regex check: does the Cypher use only known node labels and relationship types?"""
    if not cypher:
        return True, []
    violations = []
    for label_group in re.findall(r'\([\w]*:([\w:]+)[\s{)]', cypher):
        for label in label_group.split(':'):
            if label and label not in _KNOWN_NODE_LABELS:
                violations.append(f"Unknown node label: '{label}'")
    for rel_type in re.findall(r'\[[\w]*:([\w]+)[\s{*\]]', cypher):
        if rel_type not in _KNOWN_REL_TYPES:
            violations.append(f"Unknown rel type: '{rel_type}'")
    return len(violations) == 0, violations


def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter / IPython kernel."""
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


class _BenchmarkLogger:
    """
    Tee all print() output to a report file.

    Terminal: replaces sys.stdout — every write() is captured automatically.
    Jupyter:  does NOT replace sys.stdout (doing so breaks Jupyter's cell
              output routing). Instead, overrides builtins.print so every
              print() call writes to both the cell display and the file.
    """
    def __init__(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(file_path, "w", encoding="utf-8")
        self._in_jupyter = _is_jupyter()

        if self._in_jupyter:
            # Override builtins.print to tee to file without touching sys.stdout
            self._original_print = builtins.print
            _file = self._file
            _orig = self._original_print

            def _tee_print(*args, **kwargs):
                _orig(*args, **kwargs)
                sep = kwargs.get("sep", " ")
                end = kwargs.get("end", "\n")
                _file.write(sep.join(str(a) for a in args) + end)
                _file.flush()

            builtins.print = _tee_print
        else:
            self._stdout = sys.stdout
            sys.stdout = self

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        if self._in_jupyter:
            builtins.print = self._original_print
        else:
            sys.stdout = self._stdout
        self._file.close()

# Import your existing modules
from simple_rag.rag.query_handler import QueryHandler, QueryResult
from simple_rag.database.neo4j.neo4j import Neo4jDatabase
from simple_rag.rag.text2cypher import CypherTranslator


@dataclass
class _CallRecord:
    """Metrics for a single LLM call (initial or retry)."""
    call_index: int          # 0 = initial, 1+ = retry N
    prompt_text: str
    token_estimate: int
    latency_ms: float
    response: str


class _InstrumentedTranslator(CypherTranslator):
    """
    Thin subclass that wraps _invoke_llm to record per-call metrics.
    Used only when validator_mode=True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_log: List[_CallRecord] = []
        self._call_counter: int = 0

    def reset_call_log(self) -> None:
        self.call_log = []
        self._call_counter = 0

    def _invoke_llm(self, prompt_text: str) -> str:
        t0 = time.time()
        response = super()._invoke_llm(prompt_text)
        latency_ms = (time.time() - t0) * 1000
        self.call_log.append(_CallRecord(
            call_index=self._call_counter,
            prompt_text=prompt_text,
            token_estimate=len(prompt_text) // 4,
            latency_ms=latency_ms,
            response=response,
        ))
        self._call_counter += 1
        return response

@dataclass
class TestResult:
    """Stores the metrics for a single test case."""
    question_id: int
    question: str
    complexity: str
    category: str = ""
    confidence: float = 0.0
    success: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    generation_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    generated_cypher: str = ""
    expected_cypher: str = ""
    is_semantically_correct: bool = False  # Matches ground truth data
    generated_results: List[Dict] = field(default_factory=list)
    expected_results: List[Dict] = field(default_factory=list)
    llm_prompt: str = ""  # full initial LLM prompt (schema + examples + question), only populated on error
    prompt_token_estimate: int = 0  # estimated token count for the initial LLM prompt
    completion_token_estimate: int = 0  # estimated tokens in the LLM response (completion side)
    schema_token_estimate: int = 0  # estimated tokens used by the schema slice in the prompt
    validation_failures: List[str] = field(default_factory=list)  # validator rule tags that fired during retries
    retry_attempts: int = 0          # total LLM calls made (1 = no retry needed)
    final_syntax_valid: bool = False  # did the final Cypher pass validation?
    # ── New metrics ──────────────────────────────────────────────────────────
    outcome: str = ""                  # pass | empty_result | incorrect_result | error | pipeline_error | routing
    gen_record_count: int = 0
    exp_record_count: int = 0
    count_ratio: float = 0.0           # gen_count / max(exp_count, 1) — 1.0 is ideal
    schema_adherent: bool = True
    schema_violations: List[str] = field(default_factory=list)
    # ── Vector Search Evaluation ─────────────────────────────────────────────
    is_vector_question: bool = False   # True if question expects chunk IDs
    expected_chunk_ids: List[str] = field(default_factory=list)  # ground truth chunk IDs
    returned_chunk_ids: List[str] = field(default_factory=list)  # returned in order of score
    chunk_found: bool = False          # at least one expected chunk in results
    chunk_rank: int = -1               # position of first expected chunk (0-indexed, -1 if not found)
    chunk_mrr: float = 0.0             # Mean Reciprocal Rank for all expected chunks
    chunk_precision_at_k: Dict[int, float] = field(default_factory=dict)  # {k: precision}
    chunk_ndcg: float = 0.0            # Normalized Discounted Cumulative Gain

class Text2CypherBenchmark:
    """
    A professional testing suite for evaluating Text-to-Cypher models using the full RAG pipeline.
    """

    def __init__(
        self,
        test_set_path: str,
        model_name: str,
        backend: str,
        interactive: bool = False,
        openai_compatible_host: str = "localhost",
        openai_compatible_port: int = 8080,
        use_schema_injection: bool = True,
        few_shot_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        retry_module: bool = True,
        validator_mode: bool = False,
        retry_strategy: str = "full",
        embed_vector_queries: bool = False,
        use_few_shot: bool = True,
        use_entity_resolver: bool = True,
        schema_version: str = "v1",
        openrouter_fallback_model: Optional[str] = None,
        use_cypher_validator: bool = True,
    ):
        self.test_path = Path(test_set_path)
        self.neo = Neo4jDatabase()
        self.retry_module = retry_module
        self.embed_vector_queries = embed_vector_queries

        # Build extra kwargs for CypherTranslator (forwarded via QueryHandler **cypher_kwargs)
        extra_kwargs = {}
        if backend == "openai":
            extra_kwargs["openai_compatible_host"] = openai_compatible_host
            extra_kwargs["openai_compatible_port"] = openai_compatible_port
        # When retry_module=False: run validation once (to record rule violations)
        # but never send a retry prompt — max_validation_retries=1 achieves this
        # because the loop breaks immediately after the first failed attempt.
        if not retry_module:
            extra_kwargs["max_validation_retries"] = 1
        extra_kwargs["retry_strategy"] = retry_strategy
        extra_kwargs["use_few_shot"] = use_few_shot
        if openrouter_fallback_model:
            extra_kwargs["openrouter_fallback_model"] = openrouter_fallback_model
        extra_kwargs["use_cypher_validator"] = use_cypher_validator

        # Initialize the QueryHandler (Classification → Schema Slice → Cypher)
        # Only load the Nomic query embedder when embed_vector_queries=True.
        # Loading it unconditionally was causing the Jupyter cell to hang at the end
        # because HuggingFaceEmbedding keeps background threads alive that block cleanup.
        self.handler = QueryHandler(
            neo4j_driver=self.neo.driver,
            cypher_model=model_name,
            cypher_backend=backend,
            use_entity_resolver=use_entity_resolver,
            few_shot_embedding_model=few_shot_embedding_model,
            enable_query_embedding=embed_vector_queries,
            schema_version=schema_version,
            **extra_kwargs,
        )

        self.results: List[TestResult] = []
        self.backend = backend
        self.interactive = interactive
        self.incorrect_queries: List[Dict[str, Any]] = []
        self.use_schema_injection = use_schema_injection
        self.validator_mode = validator_mode

        # Upgrade the translator in-place when validator_mode is on.
        # Reassigning __class__ is safe here: _InstrumentedTranslator only adds
        # new instance attributes and overrides one method; the memory layout
        # of the base class is fully compatible.
        if validator_mode:
            self.handler.translator.__class__ = _InstrumentedTranslator
            self.handler.translator.call_log = []
            self.handler.translator._call_counter = 0

        
    def load_tests(self) -> List[Dict]:
        """Loads test cases from the JSON file."""
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test file not found at: {self.test_path}")

        with open(self.test_path, 'r') as f:
            return json.load(f)

    def _normalize_records(self, records: List[Dict]) -> List[Dict]:
        """Helper to normalize DB results for comparison (ignoring order/formatting)."""
        normalized = []
        for r in records:
            normalized.append({k: str(v) for k, v in r.items()})
        try:
            return sorted(normalized, key=lambda x: '|'.join(str(v) for v in x.values()))
        except (IndexError, TypeError):
            return normalized
    
    def _compare_results_flexible(self, gen_records: List[Dict], exp_records) -> tuple[bool, str]:
        """
        Compare results flexibly - ignores alias names, only compares values.
        Returns: (is_match, match_type) where match_type is 'exact', 'partial', or 'mismatch'
        """
        try:
            # Handle case where exp_records is a simple string/number
            if isinstance(exp_records, str):
                try:
                    exp_records = [json.loads(exp_records)]
                except json.JSONDecodeError:
                    exp_records = [{"value": exp_records}]
            
            # Extract all values from records (ignoring keys)
            def extract_all_values(records):
                all_values = set()
                if not isinstance(records, list):
                     return set([str(records)])
                for record in records:
                    if isinstance(record, (int, float, str)):
                        all_values.add(str(record))
                        continue
                    for v in record.values():
                        if isinstance(v, float):
                            all_values.add(str(round(v, 2)))
                        else:
                            all_values.add(str(v))
                return all_values

            def extract_string_entities(records):
                """Extracts only text/string values to check if the main entities match."""
                str_vals = set()
                if not isinstance(records, list): return str_vals
                for record in records:
                    if isinstance(record, (int, float)):
                        continue
                    if isinstance(record, str):
                        str_vals.add(record)
                        continue
                    for v in record.values():
                        if isinstance(v, str) and not str(v).replace('.', '', 1).isdigit():
                            str_vals.add(v)
                return str_vals
            
            # 0 generated records with non-empty expected = definitive failure
            if not gen_records and exp_records:
                return (False, 'mismatch')

            gen_values = extract_all_values(gen_records)
            exp_values = extract_all_values(exp_records)

            # Check match type
            if gen_values == exp_values:
                return (True, 'exact')
            elif exp_values.issubset(gen_values):
                return (True, 'partial')  # Generated has all expected + more
            elif gen_values.issubset(exp_values):
                return (True, 'partial')  # Generated has subset of expected
            else:
                gen_strs = extract_string_entities(gen_records)
                exp_strs = extract_string_entities(exp_records)
                
                if gen_strs and exp_strs:
                    if gen_strs.issubset(exp_strs) or exp_strs.issubset(gen_strs):
                        return (True, 'partial')
                    # Retry subset check after stripping null-like placeholders (e.g. 'N/A')
                    # so that a correct but sparser result isn't penalised for missing fields.
                    null_like = {'N/A', 'None', 'null', 'n/a', ''}
                    gen_strs_clean = gen_strs - null_like
                    if gen_strs_clean and gen_strs_clean.issubset(exp_strs):
                        return (True, 'partial')

                return (False, 'mismatch')
            
        except Exception as e:
            print(f"⚠️  Error comparing results: {e}")
            return (False, 'error')

    def _is_small_subset_match(self, gen_records: List[Dict], exp_records: List[Dict], threshold: int = 10, overlap_ratio: float = 0.6) -> bool:
        """
        Returns True if gen_records is a record-level subset of exp_records AND
        len(gen_records) <= threshold. Comparison ignores key/alias names.

        Two tiers:
        1. Strict: gen values ⊆ exp values OR exp values ⊆ gen values.
        2. Overlap fallback: the intersection of gen and exp values covers at
           least `overlap_ratio` of the expected record's values (handles cases
           where generated returns extra columns or is missing non-essential ones).
        """
        if not gen_records or len(gen_records) > threshold:
            return False

        def record_str_values(record: Dict) -> set:
            vals = set()
            for v in record.values():
                s = str(v).strip()
                if s:
                    vals.add(s)
            return vals

        exp_value_sets = [record_str_values(r) for r in exp_records]

        def matches_any(gen_vals: set) -> bool:
            for exp_vals in exp_value_sets:
                if gen_vals.issubset(exp_vals) or exp_vals.issubset(gen_vals):
                    return True
                # Overlap fallback: enough expected values are present in generated
                if exp_vals and len(gen_vals & exp_vals) / len(exp_vals) >= overlap_ratio:
                    return True
            return False

        for gen_rec in gen_records:
            if not matches_any(record_str_values(gen_rec)):
                return False
        return True

    def evaluate_single_question(self, index: int, item: Dict) -> TestResult:
        """Runs the pipeline for a single question and returns metrics."""
        question = item['question']
        expected_cypher = item.get('expected_cypher')
        expected_answer = item.get('ground_truth_answer')
        expected_chunk_ids = item.get('answer')  # Vector question: expected chunk IDs

        result = TestResult(
            question_id=index,
            question=question,
            complexity=item.get('complexity', 'unknown'),
            expected_cypher=expected_cypher,
            is_vector_question=bool(expected_chunk_ids),
            expected_chunk_ids=expected_chunk_ids if isinstance(expected_chunk_ids, list) else ([expected_chunk_ids] if expected_chunk_ids else [])
        )

        print(f"\n{'='*80}")
        print(f"📝 Q{index}: {question}")
        print(f"{'='*80}")

        # Route vector questions to specialized evaluation
        if result.is_vector_question:
            return self._evaluate_vector_question(result, item)
        
        # 1. Measure Pipeline (Classification + Translation)
        try:
            if self.validator_mode:
                self.handler.translator.reset_call_log()
            gen_start = time.time()
            handle_result = self.handler.handle(question, use_schema_injection=self.use_schema_injection)
            result.generation_time_ms = (time.time() - gen_start) * 1000
            
            result.category = handle_result.category
            result.confidence = handle_result.confidence
            result.generated_cypher = handle_result.cypher or ""
            result.llm_prompt = getattr(self.handler.translator, 'last_initial_prompt', '') or ''
            # Use cumulative token count across initial call + all retry calls
            result.prompt_token_estimate = getattr(self.handler.translator, 'last_total_prompt_tokens', 0) or (len(result.llm_prompt) // 4)
            result.completion_token_estimate = getattr(self.handler.translator, 'last_completion_tokens', 0)
            result.schema_token_estimate = getattr(self.handler.translator, 'last_schema_tokens', 0)
            result.validation_failures = list(getattr(self.handler.translator, 'last_validation_failures', []))
            result.retry_attempts = getattr(self.handler.translator, 'last_retry_attempts', 0)

            # Static schema adherence check (no DB needed)
            if result.generated_cypher:
                adherent, violations = _check_schema_adherence(result.generated_cypher)
                result.schema_adherent = adherent
                result.schema_violations = violations
                if violations:
                    print(f"⚠️  Schema violations detected: {violations}")

            if handle_result.error:
                result.error_type = f"Pipeline Error ({handle_result.category})"
                result.error_message = handle_result.error
                result.outcome = "pipeline_error"
                print(f"❌ Pipeline failed: {handle_result.error}")
                return result

            if handle_result.requires_vector_search and not handle_result.cypher:
                result.error_type = "Routing"
                result.error_message = f"Query routed to active vector search (category: {handle_result.category})"
                result.outcome = "routing"
                print(f"ℹ️  Routed to Vector Search (skipping Cypher benchmark for this item)")
                return result

        except Exception as e:
            result.error_type = "Execution Exception"
            result.error_message = str(e)
            print(f"❌ Execution error: {e}")
            return result

        # 2. Measure Execution & Accuracy
        try:
            # Build parameters — pass $queryVector if either query needs it.
            # handle() is called with execute=False so query_embedding is None by default.
            # When embed_vector_queries=True, we compute it on-demand for any query
            # whose generated or expected Cypher references $queryVector.
            query_embedding = (
                handle_result.query_embedding
            )
            needs_vector = (
                "$queryVector" in result.generated_cypher
                or "$queryVector" in (expected_cypher or "")
            )
            if query_embedding is None and needs_vector and self.embed_vector_queries:
                query_embedding = self.handler._embed_query(question)
                if query_embedding is None:
                    print("⚠️  embed_vector_queries=True but embedder returned None — skipping vector param")
            exec_params = {}
            if query_embedding is not None:
                exec_params["queryVector"] = query_embedding
                exec_params["k"] = 5

            # Run Generated Query (30s timeout to avoid hanging on expensive queries)
            exec_start = time.time()
            with self.neo.driver.session() as session:
                gen_res = list(session.run(
                    result.generated_cypher, exec_params,
                    timeout=10,
                ))
                gen_records = [r.data() for r in gen_res]
            result.execution_time_ms = (time.time() - exec_start) * 1000

            # Run Expected Query (30s timeout)
            with self.neo.driver.session() as session:
                exp_res = list(session.run(
                    expected_cypher, exec_params,
                    timeout=10,
                ))
                exp_records = [r.data() for r in exp_res]

            # Store results for comparison
            result.generated_results = gen_records
            result.expected_results = exp_records
            result.gen_record_count = len(gen_records)
            result.exp_record_count = len(exp_records)
            result.count_ratio = len(gen_records) / max(len(exp_records), 1)
            
            # Display both queries
            print(f"\n🔍 ROUTING: {result.category} ({result.confidence:.2%})")
            print(f"Generated Cypher: {result.generated_cypher}")
            print(f"Expected Cypher:  {expected_cypher}")

            # Display results
            print(f"\n📊 RESULTS:")
            print(f"Generated ({len(gen_records)} records):")
            for i, record in enumerate(gen_records[:3]):
                print(f"  [{i+1}] {record}")
            if len(gen_records) > 3:
                print(f"  ... and {len(gen_records)-3} more")
            
            print(f"Expected ({len(exp_records)} records):")
            for i, record in enumerate(exp_records[:3]):
                print(f"  [{i+1}] {record}")
            if len(exp_records) > 3:
                print(f"  ... and {len(exp_records)-3} more")
            
            # Compare Results
            is_match, match_type = self._compare_results_flexible(gen_records, exp_records)
            
            if not is_match and expected_answer:
                is_match, match_type = self._compare_results_flexible(gen_records, expected_answer)
            
            if not is_match and self._is_small_subset_match(gen_records, exp_records):
                is_match = True
                match_type = 'small_subset'

            if is_match:
                result.success = True
                result.is_semantically_correct = True
                result.outcome = "pass"
                match_label = "Exact" if match_type == 'exact' else ("Small subset" if match_type == 'small_subset' else "Partial")
                print(f"\n✅ PASS - {match_label} match")
            else:
                result.success = False
                result.error_type = "Incorrect Result"
                result.error_message = f"Got {len(gen_records)} records, expected {len(exp_records)}"
                result.outcome = "empty_result" if len(gen_records) == 0 and len(exp_records) > 0 else "incorrect_result"
                print(f"\n❌ FAIL - Result mismatch")

        except Exception as e:
            result.success = False
            result.error_type = "Syntax/Execution Error"
            result.error_message = str(e)
            result.outcome = "error"
            print(f"\n❌ FAIL - Execution error: {str(e)[:100]}")
            
        result.final_syntax_valid = (
            bool(result.generated_cypher)
            and result.error_type not in ("Syntax/Execution Error", "Pipeline Error")
            and not (result.error_type or "").startswith("Pipeline Error")
        )
        print(f"⏱️  Timings: Generation={result.generation_time_ms:.0f}ms | Execution={result.execution_time_ms:.0f}ms")
        print(f"🔢 Prompt tokens (est.): {result.prompt_token_estimate:,}")

        if self.validator_mode:
            self._print_validator_question_detail(result)

        if self.interactive:
            user_validation = self._ask_user_validation(result, item)
            if not user_validation:
                self._save_incorrect_query(result, item)
        
        return result

    def _evaluate_vector_question(self, result: TestResult, item: Dict) -> TestResult:
        """
        Evaluate a vector search question by comparing returned chunk IDs against
        expected chunk IDs. Calculates ranking metrics: rank, MRR, Precision@K, NDCG.

        Args:
            result: TestResult with is_vector_question=True and expected_chunk_ids set
            item: Test item dict with 'question' and 'answer' (expected chunk IDs)

        Returns:
            Updated TestResult with vector metrics populated
        """
        question = item['question']
        expected_chunk_ids = item.get('answer', [])
        if not isinstance(expected_chunk_ids, list):
            expected_chunk_ids = [expected_chunk_ids] if expected_chunk_ids else []

        try:
            # Generate Cypher query for vector search
            gen_start = time.time()
            handle_result = self.handler.handle(question, use_schema_injection=self.use_schema_injection)
            result.generation_time_ms = (time.time() - gen_start) * 1000

            result.category = handle_result.category
            result.confidence = handle_result.confidence
            result.generated_cypher = handle_result.cypher or ""

            if handle_result.error:
                result.error_type = "Pipeline Error"
                result.error_message = handle_result.error
                result.outcome = "pipeline_error"
                result.success = False
                print(f"❌ Pipeline failed: {handle_result.error}")
                return result

            if not result.generated_cypher:
                result.error_type = "No Cypher Generated"
                result.error_message = "Handler returned empty Cypher"
                result.outcome = "error"
                result.success = False
                print(f"❌ No Cypher generated")
                return result

            print(f"\n🔍 Vector Search Query")
            print(f"Generated Cypher:\n{result.generated_cypher}")
            print(f"\nExpected chunk IDs: {expected_chunk_ids}")

            # Execute the vector search query
            exec_start = time.time()
            query_vec = (
                handle_result.query_embedding
                or self.handler._embed_query(question)
            )
            exec_params = {
                "queryVector": query_vec,
                "k": max(10, len(expected_chunk_ids) * 2)  # Get more results for ranking
            }

            with self.neo.driver.session() as session:
                exec_result = list(session.run(
                    result.generated_cypher,
                    exec_params,
                    timeout=10
                ))
                gen_records = [r.data() for r in exec_result]
            result.execution_time_ms = (time.time() - exec_start) * 1000

            result.generated_results = gen_records
            result.gen_record_count = len(gen_records)

            # Extract chunk IDs from results (order preserved from query execution)
            returned_chunk_ids = []
            for record in gen_records:
                # Try common chunk ID field names — 'chunkId' is the canonical alias
                # used when examples include `chunk.id AS chunkId` in the RETURN clause
                chunk_id = (
                    record.get('chunkId') or
                    record.get('chunk.id') or
                    record.get('chunk_id') or
                    record.get('id') or
                    record.get('node.id')
                )
                if chunk_id:
                    returned_chunk_ids.append(chunk_id)

            result.returned_chunk_ids = returned_chunk_ids

            # Display results
            print(f"\n📊 GENERATED RESULTS: {len(gen_records)} chunks retrieved")
            for i, (record, chunk_id) in enumerate(zip(gen_records[:5], returned_chunk_ids[:5])):
                score = record.get('score', 'N/A')
                is_expected = "✓" if chunk_id in expected_chunk_ids else " "
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                print(f"  [{i+1}] {is_expected} ID: {chunk_id} | Score: {score_str}")
            if len(gen_records) > 5:
                print(f"  ... and {len(gen_records)-5} more")

            # When generated query returns nothing, run expected query for comparison
            if len(gen_records) == 0 and item.get('expected_cypher'):
                expected_cypher = item['expected_cypher']
                print(f"\n📋 COMPARISON — Running expected Cypher:")
                print(f"  Generated : {result.generated_cypher}")
                print(f"  Expected  : {expected_cypher}")
                try:
                    with self.neo.driver.session() as session:
                        exp_exec = list(session.run(expected_cypher, exec_params, timeout=10))
                        exp_records_cmp = [r.data() for r in exp_exec]
                    print(f"\n📊 EXPECTED RESULTS: {len(exp_records_cmp)} chunks retrieved")
                    for i, record in enumerate(exp_records_cmp[:5]):
                        chunk_id = (
                            record.get('chunkId') or record.get('chunk.id') or
                            record.get('chunk_id') or record.get('id') or record.get('node.id')
                        )
                        score = record.get('score', 'N/A')
                        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                        print(f"  [{i+1}] ID: {chunk_id or '?'} | Score: {score_str}")
                    if len(exp_records_cmp) > 5:
                        print(f"  ... and {len(exp_records_cmp)-5} more")
                except Exception as exp_e:
                    print(f"  ⚠️  Could not run expected query: {exp_e}")

            # Calculate metrics
            if returned_chunk_ids and expected_chunk_ids:
                self._calculate_chunk_metrics(result, expected_chunk_ids, returned_chunk_ids)

                if result.chunk_found:
                    result.success = True
                    result.outcome = "pass"
                    print(f"\n✅ PASS - Found expected chunk at rank {result.chunk_rank + 1}")
                else:
                    result.success = False
                    result.outcome = "incorrect_result"
                    result.error_type = "Expected chunk not found"
                    print(f"\n❌ FAIL - Expected chunk not found in results")
            else:
                result.success = len(returned_chunk_ids) > 0
                result.outcome = "pass" if result.success else "empty_result"
                if not result.success:
                    result.error_type = "No results returned"
                    print(f"\n❌ FAIL - No results returned")

            print(f"⏱️  Timings: Generation={result.generation_time_ms:.0f}ms | Execution={result.execution_time_ms:.0f}ms")

        except Exception as e:
            result.success = False
            result.error_type = "Execution Exception"
            result.error_message = str(e)
            result.outcome = "error"
            print(f"\n❌ FAIL - Execution error: {str(e)[:100]}")

        return result

    def _calculate_chunk_metrics(
        self,
        result: TestResult,
        expected_chunk_ids: List[str],
        returned_chunk_ids: List[str]
    ) -> None:
        """
        Calculate ranking metrics for vector search results.

        Metrics:
        - chunk_found: bool — at least one expected chunk in results
        - chunk_rank: int — 0-indexed position of first expected chunk (-1 if not found)
        - chunk_mrr: float — Mean Reciprocal Rank across all expected chunks
        - chunk_precision_at_k: dict — {k: precision@k} for k in [1, 3, 5, 10]
        - chunk_ndcg: float — Normalized Discounted Cumulative Gain
        """
        expected_set = set(expected_chunk_ids)
        returned_set = set(returned_chunk_ids)

        # Chunk Found
        result.chunk_found = bool(expected_set & returned_set)

        # Chunk Rank (position of first expected chunk)
        result.chunk_rank = -1
        for i, chunk_id in enumerate(returned_chunk_ids):
            if chunk_id in expected_set:
                result.chunk_rank = i
                break

        # Mean Reciprocal Rank (MRR)
        # Average of 1/(rank+1) for each expected chunk found
        mrr_values = []
        for chunk_id in expected_chunk_ids:
            try:
                rank = returned_chunk_ids.index(chunk_id)
                mrr_values.append(1.0 / (rank + 1))
            except ValueError:
                # Expected chunk not in returned list
                mrr_values.append(0.0)
        result.chunk_mrr = statistics.mean(mrr_values) if mrr_values else 0.0

        # Precision@K for k in [1, 3, 5, 10]
        for k in [1, 3, 5, 10]:
            top_k = set(returned_chunk_ids[:k])
            matches = len(expected_set & top_k)
            precision = matches / min(k, len(expected_chunk_ids))
            result.chunk_precision_at_k[k] = precision

        # NDCG (Normalized Discounted Cumulative Gain)
        # Ideal DCG: assume best ranking (expected chunks first)
        ideal_dcg = sum(1.0 / (i + 1) for i in range(min(len(expected_chunk_ids), len(returned_chunk_ids))))

        # Actual DCG: rank-based score for found chunks
        actual_dcg = 0.0
        for i, chunk_id in enumerate(returned_chunk_ids):
            if chunk_id in expected_set:
                actual_dcg += 1.0 / (i + 1)

        result.chunk_ndcg = (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

        print(f"\n📈 CHUNK METRICS:")
        print(f"  Found: {result.chunk_found}")
        print(f"  Rank: {result.chunk_rank + 1 if result.chunk_rank >= 0 else 'N/A'}")
        print(f"  MRR: {result.chunk_mrr:.4f}")
        print(f"  Precision@K: {result.chunk_precision_at_k}")
        print(f"  NDCG: {result.chunk_ndcg:.4f}")

    def run(self, complexity_filter: Optional[List[str]] = None):
        """Main execution loop. Saves full output to reports/ folder."""
        # Refresh entity cache so any DB changes made after initialization are picked up
        if hasattr(self.handler, 'translator') and self.handler.translator:
            self.handler.translator.refresh_entity_cache()

        test_data = self.load_tests()

        if complexity_filter is None and self.backend == "groq":
            complexity_filter = ["hard"]
            print(f"⚠️  Groq backend detected: Auto-filtering to {complexity_filter} questions only")

        if complexity_filter:
            filter_lower = [f.lower() for f in complexity_filter]
            test_data = [item for item in test_data if item.get('complexity', '').lower() in filter_lower]

        # --- Report file setup ---
        reports_dir = self.test_path.parent / "reports"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self.handler.translator.model_name.replace("/", "_").replace(":", "-")
        report_path = reports_dir / f"benchmark_{timestamp}_{model_slug}.txt"
        benchmark_logger = _BenchmarkLogger(report_path)
        # -------------------------

        print(f"\n{'='*80}")
        print(f"🚀 Text-to-Cypher Integrated Pipeline Benchmark")
        print(f"{'='*80}")
        schema_mode_label = "slice (classifier-injected)" if self.use_schema_injection else "full DETAILED_SCHEMA"
        retry_label = (f"ENABLED ({self.handler.translator.max_validation_retries} attempts)"
                       if self.retry_module else "DISABLED (1 attempt, no retry prompts)")
        retry_strategy_label = self.handler.translator.retry_strategy
        vector_label = "ON (Nomic embedder active)" if self.embed_vector_queries else "OFF (embedder not loaded — vector queries will fail)"
        few_shot_label = "ON" if self.handler.translator.use_few_shot else "OFF (no examples injected)"
        print(f"📊 Total Questions: {len(test_data)}")
        print(f"🤖 Code Model: {self.handler.translator.model_name}")
        print(f"🏷️  Classifier: SetFit (6 categories)")
        print(f"📐 Schema Mode:  {schema_mode_label} | retries always use full schema")
        print(f"🔄 Retry Module: {retry_label} | strategy: {retry_strategy_label}")
        print(f"🧬 Vector Embed: {vector_label}")
        print(f"📚 Few-Shot:     {few_shot_label}")
        print(f"📁 Report: {report_path}")
        print(f"{'='*80}\n")

        self._validator_call_logs: List[List[_CallRecord]] = []

        for i, item in enumerate(test_data, 1):
            res = self.evaluate_single_question(i, item)
            self.results.append(res)
            # Snapshot call log right after each question (before next reset)
            if self.validator_mode:
                self._validator_call_logs.append(
                    list(getattr(self.handler.translator, 'call_log', []))
                )

        self.print_report()
        benchmark_logger.close()
        # Close Neo4j and LLM resources before writing reports — avoids Jupyter
        # hanging on HuggingFace background threads during file I/O.
        self.cleanup()
        print(f"\n✅ Report saved → {report_path}")
        details_path = self._save_details_report(reports_dir, timestamp, model_slug)
        print(f"📄 Details saved → {details_path}")
        json_path = reports_dir / f"benchmark_{timestamp}_{model_slug}.json"
        self._save_json_results(json_path)
        if self.validator_mode:
            validator_path = self._save_validator_report(reports_dir, timestamp, model_slug)
            print(f"🔬 Validator report → {validator_path}")

    @staticmethod
    def _print_chunks(result: "TestResult", max_text: int = 400) -> None:
        """Pretty-print retrieved chunks with their text content."""
        records = result.generated_results
        expected_set = set(result.expected_chunk_ids)
        if not records:
            print("  (no chunks retrieved)")
            return
        for i, record in enumerate(records, 1):
            chunk_id = (
                record.get('chunkId') or record.get('chunk.id') or
                record.get('chunk_id') or record.get('id') or '?'
            )
            score = record.get('score')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            hit = " ✓" if chunk_id in expected_set else ""
            ticker = record.get('ticker', record.get('c.ticker', ''))
            year   = record.get('filingYear', record.get('r.year', ''))
            meta   = " | ".join(str(v) for v in [ticker, year] if v)
            text   = record.get('text', record.get('chunk.text', ''))
            text_snippet = text[:max_text].replace('\n', ' ').strip() if text else '(no text)'
            if len(text or '') > max_text:
                text_snippet += '…'
            print(f"  {'─'*72}")
            print(f"  [{i}]{hit} {chunk_id} | Score: {score_str}" + (f" | {meta}" if meta else ""))
            print(f"      {text_snippet}")
        print(f"  {'─'*72}")

    def run_vector(self, show_chunks: bool = True):
        """
        Run the benchmark exclusively on vector test cases (complexity='vector' /
        expected_cypher contains $queryVector) and execute each generated query
        against Neo4j.  Prints a vector-focused summary report (MRR, P@K, NDCG,
        chunk-found rate) and saves it alongside the standard reports.

        Args:
            show_chunks: If True (default), print chunk text after each question.
        """
        if hasattr(self.handler, 'translator') and self.handler.translator:
            self.handler.translator.refresh_entity_cache()

        all_tests = self.load_tests()
        vector_tests = [
            item for item in all_tests
            if item.get('complexity', '').lower() == 'vector'
            or '$queryVector' in item.get('expected_cypher', '')
        ]

        if not vector_tests:
            print("⚠️  No vector test cases found in test set.")
            return

        # ── Report setup (mirrors run()) ─────────────────────────────────────
        reports_dir = self.test_path.parent / "reports"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self.handler.translator.model_name.replace("/", "_").replace(":", "-")
        report_path = reports_dir / f"vector_{timestamp}_{model_slug}.txt"
        benchmark_logger = _BenchmarkLogger(report_path)
        # ─────────────────────────────────────────────────────────────────────

        print(f"\n{'='*80}")
        print(f"🔍 Vector Search Benchmark")
        print(f"{'='*80}")
        print(f"📊 Vector Questions: {len(vector_tests)}")
        print(f"🤖 Model: {self.handler.translator.model_name}")
        vector_label = "ON (Nomic embedder active)" if self.embed_vector_queries else "OFF (embedder not loaded — vector queries will fail)"
        print(f"🧬 Vector Embed: {vector_label}")
        print(f"📁 Report: {report_path}")
        print(f"{'='*80}\n")

        vector_results: List[TestResult] = []
        for i, item in enumerate(vector_tests, 1):
            res = self.evaluate_single_question(i, item)
            vector_results.append(res)
            if show_chunks:
                print(f"\n📄 CHUNK TEXTS ({len(res.generated_results)} retrieved):")
                self._print_chunks(res)

        # ── Vector-focused summary report ─────────────────────────────────────
        total = len(vector_results)
        passed = sum(1 for r in vector_results if r.success)
        found  = sum(1 for r in vector_results if r.chunk_found)

        mrr_vals   = [r.chunk_mrr   for r in vector_results if r.is_vector_question]
        ndcg_vals  = [r.chunk_ndcg  for r in vector_results if r.is_vector_question]
        rank_vals  = [r.chunk_rank  for r in vector_results if r.chunk_rank >= 0]

        avg_mrr  = statistics.mean(mrr_vals)  if mrr_vals  else 0.0
        avg_ndcg = statistics.mean(ndcg_vals) if ndcg_vals else 0.0
        avg_rank = statistics.mean(rank_vals) if rank_vals else float('nan')

        # Precision@K aggregated across all questions
        pk_agg: Dict[int, List[float]] = {k: [] for k in [1, 3, 5, 10]}
        for r in vector_results:
            for k, v in r.chunk_precision_at_k.items():
                pk_agg[k].append(v)
        avg_pk = {k: statistics.mean(vals) if vals else 0.0 for k, vals in pk_agg.items()}

        gen_times  = [r.generation_time_ms for r in vector_results if r.generation_time_ms > 0]
        exec_times = [r.execution_time_ms  for r in vector_results if r.execution_time_ms  > 0]
        avg_gen  = statistics.mean(gen_times)  if gen_times  else 0.0
        avg_exec = statistics.mean(exec_times) if exec_times else 0.0

        print(f"\n{'='*60}")
        print(f"📊 VECTOR BENCHMARK REPORT")
        print(f"{'='*60}")
        print(f"Total Questions:      {total}")
        print(f"Chunk Found (≥1):     {found}  ({found/total*100:.1f}%)")
        print(f"Passed (any metric):  {passed}  ({passed/total*100:.1f}%)")
        print(f"{'─'*60}")
        print(f"Mean Reciprocal Rank: {avg_mrr:.4f}")
        print(f"NDCG:                 {avg_ndcg:.4f}")
        print(f"Avg First-Hit Rank:   {avg_rank:.1f}  (1-indexed, lower is better)")
        print(f"{'─'*60}")
        print(f"Precision@1:          {avg_pk[1]:.4f}")
        print(f"Precision@3:          {avg_pk[3]:.4f}")
        print(f"Precision@5:          {avg_pk[5]:.4f}")
        print(f"Precision@10:         {avg_pk[10]:.4f}")
        print(f"{'─'*60}")
        print(f"Avg Generation Time:  {avg_gen:.0f} ms")
        print(f"Avg Execution Time:   {avg_exec:.0f} ms")
        print(f"{'='*60}")

        # Per-question failures
        failed_results = [r for r in vector_results if not r.success]
        if failed_results:
            print(f"\n📋 FAILED QUESTIONS ({len(failed_results)}):")
            for r in failed_results:
                print(f"\n  ❌ Q{r.question_id}")
                print(f"     Question : {r.question}")
                print(f"     Error    : {r.error_type} — {r.error_message}")
                print(f"     Cypher   : {r.generated_cypher or '(none)'}")
                if r.returned_chunk_ids:
                    print(f"     Returned : {r.returned_chunk_ids[:5]}")
                if r.expected_chunk_ids:
                    print(f"     Expected : {r.expected_chunk_ids}")

        benchmark_logger.close()
        self.cleanup()
        print(f"\n✅ Report saved → {report_path}")

    def _print_validator_question_detail(self, result: TestResult) -> None:
        """Print per-call retry detail for one question (validator_mode only)."""
        translator = self.handler.translator
        call_log: List[_CallRecord] = getattr(translator, 'call_log', [])
        if not call_log:
            return
        print(f"\n  🔬 VALIDATOR DETAIL — {len(call_log)} LLM call(s)")
        for rec in call_log:
            label = "initial" if rec.call_index == 0 else f"retry {rec.call_index}"
            print(f"  ┌─ Call {rec.call_index} ({label})  |  {rec.token_estimate:,} tokens  |  {rec.latency_ms:.0f} ms")
            # Show first 300 chars of prompt so it's readable without flooding output
            prompt_snippet = rec.prompt_text[:300].replace('\n', ' ')
            print(f"  │  Prompt snippet : {prompt_snippet}{'…' if len(rec.prompt_text) > 300 else ''}")
            print(f"  └─ Cypher out     : {rec.response[:200].strip()}")

    def _save_validator_report(self, reports_dir: Path, timestamp: str, model_slug: str) -> Path:
        """Write full per-call detail to reports/validator/."""
        validator_dir = reports_dir / "validator"
        validator_dir.mkdir(parents=True, exist_ok=True)
        path = validator_dir / f"validator_{timestamp}_{model_slug}.txt"

        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        all_vf = [rule for r in self.results for rule in r.validation_failures]

        # Collect per-question call logs stored on TestResult (we piggyback on
        # validation_failures order; the raw call logs are on the translator and
        # are overwritten each question, so we serialise them into the report file
        # by re-reading the translator's last state — but since we print live we
        # instead rely on the _call_log_store we build below in run()).
        call_logs: List[List[_CallRecord]] = getattr(self, '_validator_call_logs', [])

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"VALIDATOR / RETRY MODULE REPORT — {timestamp} — {model_slug}\n")
            f.write(f"Total: {total}  |  Passed: {passed}  |  Failed: {total-passed}  |  Accuracy: {passed/total*100:.2f}%\n")
            f.write("=" * 80 + "\n")
            f.write("Only questions that triggered at least one retry are shown below.\n")
            f.write("=" * 80 + "\n\n")

            for i, r in enumerate(self.results):
                log = call_logs[i] if i < len(call_logs) else []
                # Skip questions that never needed a retry (only 1 LLM call = initial)
                if len(log) <= 1:
                    continue
                status = "PASS" if r.success else "FAIL"
                retry_log = [rec for rec in log if rec.call_index > 0]
                total_q_tokens = sum(c.token_estimate for c in log)
                total_q_ms = sum(c.latency_ms for c in log)
                f.write(f"[{status}] Q{r.question_id} [{r.complexity.upper()}] — {r.category} ({r.confidence:.1%})\n")
                f.write(f"Question    : {r.question}\n")
                f.write(f"LLM calls   : {len(log)} ({len(retry_log)} retr{'y' if len(retry_log)==1 else 'ies'})  |  Total tokens: {total_q_tokens:,}  |  Total latency: {total_q_ms:.0f} ms\n")
                if r.validation_failures:
                    f.write(f"Rules fired : {', '.join(r.validation_failures)}\n")
                for rec in log:
                    label = "initial" if rec.call_index == 0 else f"retry {rec.call_index}"
                    f.write(f"\n  --- Call {rec.call_index} ({label}) ---\n")
                    f.write(f"  Tokens   : {rec.token_estimate:,}\n")
                    f.write(f"  Latency  : {rec.latency_ms:.0f} ms\n")
                    f.write(f"  Cypher   : {rec.response.strip()[:400]}\n")
                    f.write(f"  Full prompt:\n")
                    for line in rec.prompt_text.splitlines():
                        f.write(f"    {line}\n")
                f.write("-" * 80 + "\n\n")

            # Aggregate section
            f.write("=" * 80 + "\n")
            f.write("AGGREGATE VALIDATOR STATS\n")
            f.write("=" * 80 + "\n")

            never_retried = [r for r in self.results if not r.validation_failures]
            retried       = [r for r in self.results if r.validation_failures]
            fixed         = [r for r in retried if r.final_syntax_valid]
            still_broken  = [r for r in retried if not r.final_syntax_valid]

            pct = lambda n, d: f"{n/d*100:.1f}%" if d else "—"
            f.write(f"\nQuestions never needing retry : {len(never_retried):3d}  ({pct(len(never_retried), total)})\n")
            f.write(f"Questions that triggered retry : {len(retried):3d}  ({pct(len(retried), total)})\n")
            f.write(f"  Fixed by retry               : {len(fixed):3d}  ({pct(len(fixed), len(retried))} of retried)\n")
            f.write(f"  Still invalid after retry    : {len(still_broken):3d}  ({pct(len(still_broken), len(retried))} of retried)\n")

            if all_vf:
                vf_counts = Counter(all_vf)
                f.write(f"\nRule violation frequency ({len(all_vf)} total triggers):\n")
                for rule, count in vf_counts.most_common():
                    bar = "█" * max(1, round(count / len(all_vf) * 20))
                    f.write(f"  {rule:<40} {count:3d}  {bar}\n")

            all_logs_flat = [rec for log in call_logs for rec in log]
            initial_calls = [rec for rec in all_logs_flat if rec.call_index == 0]
            retry_calls   = [rec for rec in all_logs_flat if rec.call_index > 0]

            if initial_calls:
                f.write(f"\nInitial call latency  — avg: {statistics.mean(c.latency_ms for c in initial_calls):.0f} ms"
                        f"  |  max: {max(c.latency_ms for c in initial_calls):.0f} ms\n")
                f.write(f"Initial call tokens   — avg: {statistics.mean(c.token_estimate for c in initial_calls):,.0f}"
                        f"  |  max: {max(c.token_estimate for c in initial_calls):,}\n")
            if retry_calls:
                f.write(f"\nRetry call latency    — avg: {statistics.mean(c.latency_ms for c in retry_calls):.0f} ms"
                        f"  |  max: {max(c.latency_ms for c in retry_calls):.0f} ms\n")
                f.write(f"Retry call tokens     — avg: {statistics.mean(c.token_estimate for c in retry_calls):,.0f}"
                        f"  |  max: {max(c.token_estimate for c in retry_calls):,}\n")
                f.write(f"Total retry tokens    : {sum(c.token_estimate for c in retry_calls):,}\n")

        return path

    def _save_details_report(self, reports_dir: Path, timestamp: str, model_slug: str) -> Path:
        """Saves a per-question detail file to reports/details/."""
        details_dir = reports_dir / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        details_path = details_dir / f"details_{timestamp}_{model_slug}.txt"

        with open(details_path, "w", encoding="utf-8") as f:
            total = len(self.results)
            passed = sum(1 for r in self.results if r.success)
            f.write(f"BENCHMARK DETAILS — {timestamp} — {model_slug}\n")
            f.write(f"Total: {total}  |  Passed: {passed}  |  Failed: {total - passed}  |  Accuracy: {passed/total*100:.2f}%\n")
            f.write("=" * 80 + "\n\n")

            for r in self.results:
                status = "✅ PASS" if r.success else "❌ FAIL"
                f.write(f"{status}  Q{r.question_id} [{r.complexity.upper()}] — {r.category} ({r.confidence:.1%})\n")
                f.write(f"Question : {r.question}\n")
                f.write(f"Generated: {r.generated_cypher or '(none)'}\n")
                f.write(f"Expected : {r.expected_cypher or '(none)'}\n")

                # Generated results
                f.write(f"Gen Results ({len(r.generated_results)} records):\n")
                for i, rec in enumerate(r.generated_results[:10]):
                    f.write(f"  [{i+1}] {rec}\n")
                if len(r.generated_results) > 10:
                    f.write(f"  ... and {len(r.generated_results) - 10} more\n")

                # Expected results
                f.write(f"Exp Results ({len(r.expected_results)} records):\n")
                for i, rec in enumerate(r.expected_results[:10]):
                    f.write(f"  [{i+1}] {rec}\n")
                if len(r.expected_results) > 10:
                    f.write(f"  ... and {len(r.expected_results) - 10} more\n")

                if not r.success and r.error_type:
                    f.write(f"Error    : {r.error_type} — {r.error_message}\n")

                f.write("-" * 80 + "\n\n")

        return details_path

    def print_report(self):
        """Generates and prints the final metrics report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        accuracy = (passed / total) * 100 if total > 0 else 0
        
        gen_times = [r.generation_time_ms for r in self.results if r.generation_time_ms > 0]
        exec_times = [r.execution_time_ms for r in self.results if r.execution_time_ms > 0]
        
        avg_gen = statistics.mean(gen_times) if gen_times else 0
        avg_exec = statistics.mean(exec_times) if exec_times else 0

        token_estimates = [r.prompt_token_estimate for r in self.results if r.prompt_token_estimate > 0]
        avg_tokens = statistics.mean(token_estimates) if token_estimates else 0
        total_tokens = sum(token_estimates)

        completion_estimates = [r.completion_token_estimate for r in self.results if r.completion_token_estimate > 0]
        avg_completion = statistics.mean(completion_estimates) if completion_estimates else 0
        total_completion = sum(completion_estimates)

        schema_estimates = [r.schema_token_estimate for r in self.results if r.schema_token_estimate > 0]
        avg_schema = statistics.mean(schema_estimates) if schema_estimates else 0

        # ESR: queries that executed without a syntax/runtime Neo4j error
        error_count = sum(1 for r in self.results if r.outcome == "error")
        esr_n = total - error_count
        esr = esr_n / total * 100 if total > 0 else 0.0

        retry_label = (f"ENABLED ({self.handler.translator.max_validation_retries} attempts)"
                       if self.retry_module else "DISABLED (no retry prompts)")
        print("\n" + "="*60)
        print("📊 PIPELINE BENCHMARK REPORT")
        print("="*60)
        print(f"Retry Module:         {retry_label}")
        print(f"Total Questions:      {total}")
        print(f"Passed (Correct Data): {passed}")
        print(f"Failed:               {failed}")
        print(f"ACCURACY (EX):        {accuracy:.2f}%")
        print(f"ESR:                  {esr:.2f}%  ({esr_n}/{total} executed without error)")
        print("-" * 60)
        print(f"Avg Dispatch Time:    {avg_gen:.2f} ms (Classify + Schema Slice + Translate)")
        print(f"Avg Execution Time:   {avg_exec:.2f} ms")
        print(f"Avg Total Latency:    {avg_gen + avg_exec:.2f} ms")
        print("-" * 60)
        print(f"Avg Prompt Tokens:    {avg_tokens:,.0f} (est.) | Total: {total_tokens:,}")
        print(f"  └─ Schema slice:    {avg_schema:,.0f} (est.) | {avg_schema/avg_tokens*100:.1f}% of prompt" if avg_tokens > 0 else "  └─ Schema slice:    n/a")
        print(f"Avg Completion Tokens:{avg_completion:,.0f} (est.) | Total: {total_completion:,}" if avg_completion > 0 else "Avg Completion Tokens: n/a (backend does not report)")
        print("="*60)

        # First-attempt accuracy (no validation retries triggered)
        first_attempt_passed = sum(1 for r in self.results if r.success and not r.validation_failures)
        first_attempt_accuracy = first_attempt_passed / total * 100 if total else 0.0
        print(f"\nFirst-attempt accuracy: {first_attempt_accuracy:.2f}%  ({first_attempt_passed}/{total} passed without retry)")

        # Count ratio
        ratios = [r.count_ratio for r in self.results if r.outcome not in ("routing", "pipeline_error", "error", "")]
        avg_ratio = statistics.mean(ratios) if ratios else 0.0
        print(f"Avg count ratio (gen/exp records): {avg_ratio:.2f}  (1.0 = ideal)")

        # Schema adherence
        with_cypher = [r for r in self.results if r.generated_cypher]
        adherent = sum(1 for r in with_cypher if r.schema_adherent)
        schema_rate = adherent / len(with_cypher) * 100 if with_cypher else 0.0
        total_violations = sum(len(r.schema_violations) for r in self.results)
        print(f"Schema adherence rate: {schema_rate:.2f}%  ({total_violations} total violations)")

        # Outcome breakdown
        outcome_counts = Counter(r.outcome for r in self.results if r.outcome)
        print(f"\n{'─'*60}")
        print("🔎 OUTCOME BREAKDOWN")
        print(f"{'─'*60}")
        for outcome, count in sorted(outcome_counts.items()):
            pct = count / total * 100
            print(f"  {outcome:<30} {count:3d}  ({pct:.1f}%)")

        # Add category breakdown with latency
        print("\n📊 CATEGORY DISPATCH SUMMARY:")
        print(f"  {'Category':<22} {'Queries':>7}  {'Accuracy':>9}  {'Avg Dispatch':>13}  {'Avg Exec':>9}")
        print(f"  {'─'*22}  {'─'*7}  {'─'*9}  {'─'*13}  {'─'*9}")
        categories = Counter([r.category for r in self.results])
        for cat, count in categories.items():
            cat_results = [r for r in self.results if r.category == cat]
            cat_passed = sum(1 for r in cat_results if r.success)
            cat_acc = (cat_passed / len(cat_results)) * 100
            cat_gen = [r.generation_time_ms for r in cat_results if r.generation_time_ms > 0]
            cat_exec = [r.execution_time_ms for r in cat_results if r.execution_time_ms > 0]
            cat_avg_gen = statistics.mean(cat_gen) if cat_gen else 0.0
            cat_avg_exec = statistics.mean(cat_exec) if cat_exec else 0.0
            print(f"  {cat:<22}  {count:>7}  {cat_acc:>8.1f}%  {cat_avg_gen:>11.0f}ms  {cat_avg_exec:>7.0f}ms")

        # Per-category failure mode breakdown
        all_cats = sorted(set(r.category for r in self.results if r.category))
        if all_cats:
            print(f"\n{'─'*60}")
            print("🔬 FAILURE MODE BY CATEGORY")
            outcome_cols = ["pass", "empty_result", "incorrect_result", "error", "pipeline_error", "routing"]
            print(f"  {'Category':<22}" + "".join(f"{o:<16}" for o in outcome_cols))
            print(f"  {'─'*20}" + "─" * (16 * len(outcome_cols)))
            for cat in all_cats:
                cat_results = [r for r in self.results if r.category == cat]
                cat_outcomes = Counter(r.outcome for r in cat_results if r.outcome)
                print(f"  {cat:<22}" + "".join(f"{cat_outcomes.get(o, 0):<16}" for o in outcome_cols))

        if failed > 0:
            print("\n📋 ERROR LOG (Failed Questions):")
            for r in self.results:
                if not r.success:
                    print(f"\n❌ Q{r.question_id} [{r.complexity.upper()}] | Cat: {r.category} ({r.confidence:.1%})")
                    print(f"   Q: {r.question}")
                    print(f"   Error: {r.error_type} - {r.error_message}")
                    if r.generated_cypher:
                        print(f"   Gen Cypher: {r.generated_cypher}")
                    if r.llm_prompt:
                        print(f"   {'─'*76}")
                        print(f"   LLM PROMPT — {r.prompt_token_estimate:,} tokens (est.) — initial request sent to model:")
                        print(f"   {'─'*76}")
                        
        self._print_failure_analytics()

        # --- NULL EXPECTED RESULTS ---
        # Questions where the expected Cypher itself returned 0 records.
        # These are untestable and must be fixed in test_set.json.
        null_expected = [
            r for r in self.results
            if hasattr(r, 'expected_results') and r.expected_results == []
            and r.expected_cypher  # has a cypher query (not skipped)
        ]
        print(f"\n{'='*60}")
        print(f"⚠️  NULL EXPECTED RESULTS — {len(null_expected)} question(s) need fixing in test_set.json")
        print(f"{'='*60}")
        if null_expected:
            for r in null_expected:
                status = "✅ PASS" if r.success else "❌ FAIL"
                print(f"\n  {status} Q{r.question_id} [{r.complexity.upper()}]: {r.question}")
                print(f"    Expected Cypher: {r.expected_cypher}")
        else:
            print("  ✅ All expected Cyphers returned results — no fixes needed.")

    def _print_failure_analytics(self):
        """Print a structured breakdown of all failure types to guide debugging."""
        failed = [r for r in self.results if not r.success]
        if not failed:
            print("\n✅ No failures — analytics not needed.")
            return

        total_failed = len(failed)

        print(f"\n{'='*60}")
        print(f"🔬 FAILURE ANALYTICS  ({total_failed} total failures)")
        print(f"{'='*60}")

        # ── 0. Validator rule violations across ALL queries (retries included) ──
        all_vf = [rule for r in self.results for rule in r.validation_failures]
        retry_note = ("each retry attempt counted separately"
                      if self.retry_module else "retry module DISABLED — first attempt only")
        if all_vf:
            vf_counts = Counter(all_vf)
            total_vf  = len(all_vf)
            queries_with_vf = sum(1 for r in self.results if r.validation_failures)
            print(f"\n🛡️  Validator Rule Violations  "
                  f"({total_vf} triggers across {queries_with_vf} queries | {retry_note}):")
            for rule, count in vf_counts.most_common():
                pct = count / total_vf * 100
                bar = "█" * max(1, round(pct / 5))
                print(f"  {rule:<35} {count:3d}  ({pct:5.1f}%)  {bar}")
        else:
            print(f"\n🛡️  Validator Rule Violations: none  [{retry_note}]")

        # ── 1. Top-level error type distribution ─────────────────────
        print(f"\n📌 Failure Type Distribution:")
        error_type_counts = Counter(r.error_type or "Unknown" for r in failed)
        for etype, count in error_type_counts.most_common():
            pct = count / total_failed * 100
            bar = "█" * max(1, round(pct / 5))
            print(f"  {etype:<40} {count:3d}  ({pct:5.1f}%)  {bar}")

        # ── 2. Incorrect Result sub-analysis ─────────────────────────
        incorrect = [r for r in failed if r.error_type == "Incorrect Result"]
        if incorrect:
            empty_gen   = [r for r in incorrect if not r.generated_results]
            nonempty    = [r for r in incorrect if r.generated_results]
            wrong_count = [r for r in nonempty
                           if len(r.generated_results) != len(r.expected_results)]
            same_count  = [r for r in nonempty
                           if len(r.generated_results) == len(r.expected_results)]
            print(f"\n📊 Incorrect Result Sub-Analysis  ({len(incorrect)} queries):")
            print(f"  Generated 0 records (empty result)  : {len(empty_gen):3d}")
            print(f"  Wrong record count                   : {len(wrong_count):3d}")
            print(f"  Same count, wrong values             : {len(same_count):3d}")
            if empty_gen:
                print(f"\n  Queries that returned 0 records:")
                for r in empty_gen:
                    q_preview = r.question[:65] + ("…" if len(r.question) > 65 else "")
                    print(f"    Q{r.question_id:<4} [{r.complexity:<6}] {q_preview}")

        # ── 3. Syntax/Execution Error — Neo4j exception types ────────
        syntax_errors = [r for r in failed if r.error_type == "Syntax/Execution Error"]
        if syntax_errors:
            print(f"\n🔎 Syntax/Execution Error Breakdown  ({len(syntax_errors)} queries):")
            exc_types: Counter = Counter()
            for r in syntax_errors:
                msg = r.error_message or ""
                # Pattern: "ExceptionName: ..." (Python exception str representation)
                m = re.match(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))', msg)
                if m:
                    exc_types[m.group(1)] += 1
                else:
                    # Neo4j wire format: "{code: Neo.ClientError.Statement.SyntaxError} ..."
                    m2 = re.search(r'Neo\.[A-Za-z.]+', msg)
                    exc_types[m2.group(0) if m2 else "Other/Unknown"] += 1
            for exc, count in exc_types.most_common():
                print(f"  {exc:<50}: {count:3d}")

        # ── 4. Pipeline Error — embedded category breakdown ──────────
        pipeline_errors = [r for r in failed
                           if r.error_type and r.error_type.startswith("Pipeline Error")]
        if pipeline_errors:
            print(f"\n⚙️  Pipeline Error Breakdown  ({len(pipeline_errors)} queries):")
            pe_cats: Counter = Counter()
            for r in pipeline_errors:
                # error_type is "Pipeline Error (category)"
                m = re.search(r'\((.+)\)', r.error_type or "")
                pe_cats[m.group(1) if m else "unknown"] += 1
            for cat, count in pe_cats.most_common():
                print(f"  {cat:<30}: {count:3d}")
            # Show the distinct error messages to understand root causes
            msg_counts: Counter = Counter(r.error_message or "" for r in pipeline_errors)
            print(f"  Root-cause messages:")
            for msg, count in msg_counts.most_common():
                print(f"    [{count}x] {msg[:80]}")

        # ── 5. Failures by complexity ─────────────────────────────────
        print(f"\n📈 Failures by Complexity:")
        for complexity in sorted(set(r.complexity for r in self.results)):
            total_c  = sum(1 for r in self.results if r.complexity == complexity)
            failed_c = sum(1 for r in failed    if r.complexity == complexity)
            pct = failed_c / total_c * 100 if total_c else 0
            bar = "█" * max(1, round(pct / 5))
            print(f"  {complexity:<10}: {failed_c:3d} / {total_c:3d} failed  ({pct:5.1f}%)  {bar}")

        # ── 6. Retry module effectiveness ─────────────────────────────
        pct = lambda n, d: f"{n/d*100:.1f}%" if d else "—"
        retried     = [r for r in self.results if r.validation_failures]
        not_retried = [r for r in self.results if not r.validation_failures]
        syntax_fixed  = [r for r in retried if r.final_syntax_valid]
        correct_after = [r for r in retried if r.success]
        still_broken  = [r for r in retried if not r.final_syntax_valid]
        no_retry_correct = [r for r in not_retried if r.success]

        print(f"\n🔄 RETRY MODULE EFFECTIVENESS  ({len(retried)} / {len(self.results)} questions triggered retries):")
        print(f"  Final syntax valid after retry  : {len(syntax_fixed):3d}  ({pct(len(syntax_fixed), len(retried))} of retried)")
        print(f"    └─ Also correct results        : {len(correct_after):3d}  ({pct(len(correct_after), len(syntax_fixed))} of syntax-valid)")
        print(f"  Still syntax broken after retry  : {len(still_broken):3d}  ({pct(len(still_broken), len(retried))} of retried)")
        print(f"  Questions never needed retry     : {len(not_retried):3d}  ({pct(len(not_retried), len(self.results))} of total)")
        print(f"    └─ Correct on first attempt    : {len(no_retry_correct):3d}  ({pct(len(no_retry_correct), len(not_retried))} of no-retry)")

    def _ask_user_validation(self, result: TestResult, item: Dict) -> bool:
        """Ask user if the generated query and results are correct."""
        print("\n🔍 USER VALIDATION REQUIRED")
        while True:
            response = input("\n❓ Correct? (y/n/s): ").strip().lower()
            if response == 'y': return True
            if response == 'n': return False
            if response == 's': return True

    def _save_incorrect_query(self, result: TestResult, item: Dict):
        self.incorrect_queries.append({
            "question_id": result.question_id,
            "question": result.question,
            "category": result.category,
            "expected_cypher": result.expected_cypher,
            "generated_cypher": result.generated_cypher,
            "error_type": result.error_type,
            "error_message": result.error_message
        })

    @staticmethod
    def _json_serializable(obj):
        """Convert Neo4j and other non-JSON-serializable types to strings."""
        # neo4j.time types: Date, DateTime, Time, LocalDateTime, LocalTime, Duration
        type_name = type(obj).__name__
        if type_name in ("Date", "DateTime", "Time", "LocalDateTime", "LocalTime", "Duration",
                         "Neo4jDateTime", "Neo4jDate", "Neo4jTime"):
            return str(obj)
        if hasattr(obj, "isoformat"):   # datetime.date / datetime.datetime
            return obj.isoformat()
        if hasattr(obj, "__str__"):
            return str(obj)
        raise TypeError(f"Object of type {type_name} is not JSON serializable")

    def _save_json_results(self, json_path: Path) -> None:
        """Save structured results to JSON for downstream use (e.g. llm_judge.py)."""
        records = []
        for r in self.results:
            records.append({
                "question_id": r.question_id,
                "question": r.question,
                "complexity": r.complexity,
                "category": r.category,
                "confidence": round(r.confidence, 4),
                "outcome": r.outcome,
                "success": r.success,
                "retry_attempts": r.retry_attempts,
                "validation_failures": r.validation_failures,
                "generated_cypher": r.generated_cypher,
                "expected_cypher": r.expected_cypher,
                "generated_results": r.generated_results[:20],
                "expected_results": r.expected_results[:20],
                "gen_record_count": r.gen_record_count,
                "exp_record_count": r.exp_record_count,
                "count_ratio": round(r.count_ratio, 3),
                "schema_adherent": r.schema_adherent,
                "schema_violations": r.schema_violations,
                "error_type": r.error_type,
                "error_message": r.error_message,
                "generation_time_ms": round(r.generation_time_ms, 1),
                "execution_time_ms": round(r.execution_time_ms, 1),
                "prompt_token_estimate": r.prompt_token_estimate,
                # Placeholder for llm_judge.py output
                "judge_verdict": None,
                "judge_reasoning": None,
            })
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=self._json_serializable)
        print(f"📄 JSON results saved → {json_path}")

    def debug(
        self,
        questions: "int | str | list[int | str | dict]",
        expected_cyphers: "list[str] | None" = None,
    ) -> None:
        """
        Run the full pipeline on one or more questions and save the complete
        debug output — including the raw LLM prompt — to:
            reports/single_question/debug_<model_slug>.txt
        The file is overwritten on every call so you always see the latest run.

        Args:
            questions:
                - int       → 1-based index into the test_set.json file
                - str       → raw question text (no expected Cypher)
                - list of any mix of the above, or dicts with keys
                  'question' and optionally 'expected_cypher'
            expected_cyphers:
                Optional parallel list of expected Cypher strings (only used
                when `questions` is a list of plain strings).

        Usage in Jupyter:
            benchmark.debug(124)                         # test_set question #124
            benchmark.debug([10, 42, 124])               # three test_set questions
            benchmark.debug("What is the expense ratio of VTI?")
            benchmark.debug([
                "What is the expense ratio of VTI?",
                "Show me the risk factors for VXUS.",
            ])
            benchmark.debug([
                {"question": "What is the expense ratio of VTI?",
                 "expected_cypher": "MATCH (f:Fund {ticker: 'VTI'})..."},
            ])
        """
        # Normalise scalar to list
        if not isinstance(questions, list):
            questions = [questions]

        # Load test set lazily (only if any index is requested)
        _test_data: "list[dict] | None" = None

        def _get_test_item(idx: int) -> dict:
            nonlocal _test_data
            if _test_data is None:
                _test_data = self.load_tests()
            if idx < 1 or idx > len(_test_data):
                raise IndexError(
                    f"Question index {idx} is out of range "
                    f"(test set has {len(_test_data)} questions, 1-based)."
                )
            item = dict(_test_data[idx - 1])   # copy so we don't mutate the original
            item.setdefault("complexity", "debug")
            return item

        items: list[dict] = []
        for i, q in enumerate(questions):
            if isinstance(q, int):
                items.append(_get_test_item(q))
            elif isinstance(q, dict):
                items.append(q)
            else:
                item = {"question": q, "complexity": "debug"}
                if expected_cyphers and i < len(expected_cyphers):
                    item["expected_cypher"] = expected_cyphers[i]
                items.append(item)

        # Output path — fixed name so it overwrites on every call
        reports_dir = self.test_path.parent / "reports" / "single_question"
        model_slug = self.handler.translator.model_name.replace("/", "_").replace(":", "-")
        report_path = reports_dir / f"debug_{model_slug}.txt"
        logger = _BenchmarkLogger(report_path)

        print(f"\n{'='*80}")
        print(f"🔬 Debug Run — {len(items)} question(s)")
        print(f"🤖 Model: {self.handler.translator.model_name}")
        print(f"📁 Output: {report_path}")
        print(f"{'='*80}\n")

        self.handler.translator._capture_prompts = True
        try:
            for i, (q_input, item) in enumerate(zip(questions, items), 1):
                if isinstance(q_input, int):
                    print(f"[test_set #{q_input}]")
                res = self.evaluate_single_question(i, item)

                # Print the full LLM prompt(s) so they appear in the saved file
                prompt = getattr(self.handler.translator, 'last_initial_prompt', '') or ''
                if prompt:
                    print(f"\n{'─'*80}")
                    print("📄 FULL LLM PROMPT — attempt 1 (initial):")
                    print('─'*80)
                    print(prompt)
                    print('─'*80)

                retry_prompts = getattr(self.handler.translator, 'last_retry_prompts', []) or []
                for r_idx, r_prompt in enumerate(retry_prompts, 2):
                    print(f"\n{'─'*80}")
                    print(f"📄 FULL LLM PROMPT — attempt {r_idx} (retry {r_idx - 1}):")
                    print('─'*80)
                    print(r_prompt)
                    print('─'*80)

                # Summary line
                outcome_icon = "✅" if res.outcome in ("pass", "partial") else "❌"
                print(f"\n{outcome_icon} Outcome: {res.outcome}")
                if res.generated_cypher:
                    print(f"   Generated : {res.generated_cypher}")
                if res.expected_cypher:
                    print(f"   Expected  : {res.expected_cypher}")
                print(f"   Prompt tokens (est.): {res.prompt_token_estimate}")

        finally:
            self.handler.translator._capture_prompts = False

        logger.close()
        print(f"\n✅ Debug output saved → {report_path}")

    def cleanup(self):
        self.handler.translator.stop_ollama_server()
        self.neo.close()

from collections import Counter