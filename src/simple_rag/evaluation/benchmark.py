import re
import sys
import builtins
import time
import json
import statistics
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


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
    validation_failures: List[str] = field(default_factory=list)  # validator rule tags that fired during retries
    retry_attempts: int = 0          # total LLM calls made (1 = no retry needed)
    final_syntax_valid: bool = False  # did the final Cypher pass validation?

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
    ):
        self.test_path = Path(test_set_path)
        self.neo = Neo4jDatabase()
        self.retry_module = retry_module

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

        # Initialize the QueryHandler (Classification → Schema Slice → Cypher)
        self.handler = QueryHandler(
            neo4j_driver=self.neo.driver,
            cypher_model=model_name,
            cypher_backend=backend,
            use_entity_resolver=True,
            few_shot_embedding_model=few_shot_embedding_model,
            **extra_kwargs,
        )

        self.results: List[TestResult] = []
        self.backend = backend
        self.interactive = interactive
        self.incorrect_queries: List[Dict[str, Any]] = []
        self.use_schema_injection = use_schema_injection

        
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
        
        result = TestResult(
            question_id=index,
            question=question,
            complexity=item.get('complexity', 'unknown'),
            expected_cypher=expected_cypher
        )

        print(f"\n{'='*80}")
        print(f"📝 Q{index}: {question}")
        print(f"{'='*80}")
        
        # 1. Measure Pipeline (Classification + Translation)
        try:
            gen_start = time.time()
            handle_result = self.handler.handle(question, use_schema_injection=self.use_schema_injection)
            result.generation_time_ms = (time.time() - gen_start) * 1000
            
            result.category = handle_result.category
            result.confidence = handle_result.confidence
            result.generated_cypher = handle_result.cypher or ""
            result.llm_prompt = getattr(self.handler.translator, 'last_initial_prompt', '') or ''
            # Use cumulative token count across initial call + all retry calls
            result.prompt_token_estimate = getattr(self.handler.translator, 'last_total_prompt_tokens', 0) or (len(result.llm_prompt) // 4)
            result.validation_failures = list(getattr(self.handler.translator, 'last_validation_failures', []))
            result.retry_attempts = getattr(self.handler.translator, 'last_retry_attempts', 0)

            if handle_result.error:
                result.error_type = f"Pipeline Error ({handle_result.category})"
                result.error_message = handle_result.error
                print(f"❌ Pipeline failed: {handle_result.error}")
                return result

            if handle_result.requires_vector_search and not handle_result.cypher:
                result.error_type = "Routing"
                result.error_message = f"Query routed to active vector search (category: {handle_result.category})"
                print(f"ℹ️  Routed to Vector Search (skipping Cypher benchmark for this item)")
                return result

        except Exception as e:
            result.error_type = "Execution Exception"
            result.error_message = str(e)
            print(f"❌ Execution error: {e}")
            return result

        # 2. Measure Execution & Accuracy
        try:
            # Build parameters — pass $queryVector if either query needs it
            query_embedding = handle_result.query_embedding
            exec_params = {}
            if query_embedding is not None:
                exec_params["queryVector"] = query_embedding
                exec_params["k"] = 5

            # Run Generated Query
            exec_start = time.time()
            with self.neo.driver.session() as session:
                gen_res = list(session.run(result.generated_cypher, exec_params))
                gen_records = [r.data() for r in gen_res]
            result.execution_time_ms = (time.time() - exec_start) * 1000

            # Run Expected Query
            with self.neo.driver.session() as session:
                exp_res = list(session.run(expected_cypher, exec_params))
                exp_records = [r.data() for r in exp_res]

            # Store results for comparison
            result.generated_results = gen_records
            result.expected_results = exp_records
            
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
                match_label = "Exact" if match_type == 'exact' else ("Small subset" if match_type == 'small_subset' else "Partial")
                print(f"\n✅ PASS - {match_label} match")
            else:
                result.success = False
                result.error_type = "Incorrect Result"
                result.error_message = f"Got {len(gen_records)} records, expected {len(exp_records)}"
                print(f"\n❌ FAIL - Result mismatch")
                
        except Exception as e:
            result.success = False
            result.error_type = "Syntax/Execution Error"
            result.error_message = str(e)
            print(f"\n❌ FAIL - Execution error: {str(e)[:100]}")
            
        result.final_syntax_valid = (
            bool(result.generated_cypher)
            and result.error_type not in ("Syntax/Execution Error", "Pipeline Error")
            and not (result.error_type or "").startswith("Pipeline Error")
        )
        print(f"⏱️  Timings: Generation={result.generation_time_ms:.0f}ms | Execution={result.execution_time_ms:.0f}ms")
        print(f"🔢 Prompt tokens (est.): {result.prompt_token_estimate:,}")
        
        if self.interactive:
            user_validation = self._ask_user_validation(result, item)
            if not user_validation:
                self._save_incorrect_query(result, item)
        
        return result

    def run(self, complexity_filter: Optional[List[str]] = None):
        """Main execution loop. Saves full output to reports/ folder."""
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
        print(f"📊 Total Questions: {len(test_data)}")
        print(f"🤖 Code Model: {self.handler.translator.model_name}")
        print(f"🏷️  Classifier: SetFit (9 categories)")
        print(f"📐 Schema Mode:  {schema_mode_label} | retries always use full schema")
        print(f"🔄 Retry Module: {retry_label}")
        print(f"📁 Report: {report_path}")
        print(f"{'='*80}\n")

        for i, item in enumerate(test_data, 1):
            res = self.evaluate_single_question(i, item)
            self.results.append(res)

        self.print_report()
        benchmark_logger.close()
        print(f"\n✅ Report saved → {report_path}")
        self.cleanup()

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

        retry_label = (f"ENABLED ({self.handler.translator.max_validation_retries} attempts)"
                       if self.retry_module else "DISABLED (no retry prompts)")
        print("\n" + "="*60)
        print("📊 PIPELINE BENCHMARK REPORT")
        print("="*60)
        print(f"Retry Module:         {retry_label}")
        print(f"Total Questions:      {total}")
        print(f"Passed (Correct Data): {passed}")
        print(f"Failed:               {failed}")
        print(f"ACCURACY:             {accuracy:.2f}%")
        print("-" * 60)
        print(f"Avg Dispatch Time:    {avg_gen:.2f} ms (Classify + Schema Slice + Translate)")
        print(f"Avg Execution Time:   {avg_exec:.2f} ms")
        print(f"Avg Prompt Tokens:    {avg_tokens:,.0f} (est.) | Total: {total_tokens:,}")
        print("="*60)
        
        # Add category breakdown
        print("\n� CATEGORY DISPATCH SUMMARY:")
        categories = Counter([r.category for r in self.results])
        for cat, count in categories.items():
            cat_results = [r for r in self.results if r.category == cat]
            cat_passed = sum(1 for r in cat_results if r.success)
            cat_acc = (cat_passed / len(cat_results)) * 100
            print(f"  - {cat:20}: {count:3} queries | Accuracy: {cat_acc:6.1f}%")

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

    def cleanup(self):
        self.handler.translator.stop_ollama_server()
        self.neo.close()

from collections import Counter