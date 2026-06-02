import sys
import re
import time
import json
import statistics
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter

# Allow running as a script — add src/ to path so `simple_rag` resolves
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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
    # Node labels: (:Label), (var:Label), (:Label:Label2)
    for label_group in re.findall(r'\([\w]*:([\w:]+)[\s{)]', cypher):
        for label in label_group.split(':'):
            if label and label not in _KNOWN_NODE_LABELS:
                violations.append(f"Unknown node label: '{label}'")
    # Relationship types: [:TYPE], [var:TYPE], [*..TYPE]
    for rel_type in re.findall(r'\[[\w]*:([\w]+)[\s{*\]]', cypher):
        if rel_type not in _KNOWN_REL_TYPES:
            violations.append(f"Unknown rel type: '{rel_type}'")
    return len(violations) == 0, violations


class _BenchmarkLogger:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()


from simple_rag.rag.query_handler import QueryHandler
from simple_rag.database.neo4j.neo4j import Neo4jDatabase


@dataclass
class TestResult:
    """Stores the metrics for a single test case."""
    question_id: int
    question: str
    complexity: str
    category: str = ""
    confidence: float = 0.0
    active_labels: List[str] = field(default_factory=list)        # all labels above threshold
    per_label_confidence: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    generation_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    generated_cypher: str = ""
    expected_cypher: str = ""
    is_semantically_correct: bool = False
    generated_results: List[Dict] = field(default_factory=list)
    expected_results: List[Dict] = field(default_factory=list)
    llm_prompt: str = ""
    prompt_token_estimate: int = 0
    schema_mode: str = ""              # "detailed" or "sliced"
    schema_name: str = ""              # slice name (e.g. "fund_basic" or "fund_basic + fund_portfolio")
    schema_text: str = ""              # actual schema text passed to LLM (truncated for display)
    attempt_count: int = 0             # number of LLM calls (1 = no retry, >1 = retried)
    needed_retry: bool = False         # convenience flag
    # ── New metrics ──────────────────────────────────────────────────────────
    outcome: str = ""                  # pass | empty_result | incorrect_result | error | pipeline_error | routing
    gen_record_count: int = 0
    exp_record_count: int = 0
    count_ratio: float = 0.0           # gen_count / max(exp_count, 1) — 1.0 is ideal
    schema_adherent: bool = True
    schema_violations: List[str] = field(default_factory=list)


@dataclass
class PipelineStats:
    """Aggregated statistics for one pipeline run (detailed or sliced)."""
    name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    accuracy: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    avg_prompt_tokens: float = 0.0
    total_prompt_tokens: int = 0
    retry_count: int = 0               # number of queries that needed at least one retry
    retry_rate: float = 0.0            # percentage of queries that needed a retry
    avg_attempts: float = 0.0          # average # of LLM calls per query
    category_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    complexity_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # ── New metrics ──────────────────────────────────────────────────────────
    first_attempt_passed: int = 0
    first_attempt_accuracy: float = 0.0
    outcome_breakdown: Dict[str, int] = field(default_factory=dict)
    avg_count_ratio: float = 0.0
    schema_adherence_rate: float = 0.0
    schema_violations_total: int = 0
    per_category_outcomes: Dict[str, Dict[str, int]] = field(default_factory=dict)


class Text2CypherBenchmark:
    """
    Benchmark suite that runs the Text-to-Cypher pipeline TWICE on the same test set:
      1. With the full DETAILED schema (use_schema_injection=False)
      2. With SCHEMA SLICING (use_schema_injection=True)
    Then prints a side-by-side comparison report.
    """

    def __init__(
        self,
        test_set_path: str,
        model_name: str,
        backend: str,
        interactive: bool = False,
        openai_compatible_host: str = "localhost",
        openai_compatible_port: int = 1234,
        use_schema_injection: bool = True,  # kept for backward-compat
        mode: str = "both",                 # "both" | "detailed" | "sliced"
    ):
        self.test_path = Path(test_set_path)
        self.model_name = model_name
        self.backend = backend
        self.interactive = interactive
        self.openai_compatible_host = openai_compatible_host
        self.openai_compatible_port = openai_compatible_port
        self.mode = mode

        self.neo = Neo4jDatabase()

        # Build extra kwargs forwarded to CypherTranslator via QueryHandler
        extra_kwargs = {}
        if backend == "openai":
            extra_kwargs["openai_compatible_host"] = openai_compatible_host
            extra_kwargs["openai_compatible_port"] = openai_compatible_port

        # ONE QueryHandler — reused for both passes (detailed + sliced)
        self.handler = QueryHandler(
            neo4j_driver=self.neo.driver,
            cypher_model=model_name,
            cypher_backend=backend,
            use_entity_resolver=True,
            **extra_kwargs,
        )

        # ── Wrap translator._invoke_llm to count attempts per translate() call ──
        self._llm_call_count = 0
        original_invoke = self.handler.translator._invoke_llm

        def counting_invoke(prompt_text: str) -> str:
            self._llm_call_count += 1
            return original_invoke(prompt_text)

        self.handler.translator._invoke_llm = counting_invoke

        self.detailed_results: List[TestResult] = []
        self.sliced_results: List[TestResult] = []
        self.incorrect_queries: List[Dict[str, Any]] = []

    def load_tests(self) -> List[Dict]:
        """Loads test cases from the JSON file."""
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test file not found at: {self.test_path}")
        with open(self.test_path, 'r') as f:
            return json.load(f)

    def _normalize_records(self, records: List[Dict]) -> List[Dict]:
        normalized = []
        for r in records:
            normalized.append({k: str(v) for k, v in r.items()})
        try:
            return sorted(normalized, key=lambda x: '|'.join(str(v) for v in x.values()))
        except (IndexError, TypeError):
            return normalized

    def _compare_results_flexible(self, gen_records: List[Dict], exp_records) -> tuple[bool, str]:
        try:
            if isinstance(exp_records, str):
                try:
                    exp_records = [json.loads(exp_records)]
                except json.JSONDecodeError:
                    exp_records = [{"value": exp_records}]

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
                str_vals = set()
                if not isinstance(records, list):
                    return str_vals
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

            if gen_values == exp_values:
                return (True, 'exact')
            elif exp_values.issubset(gen_values):
                return (True, 'partial')
            elif gen_values.issubset(exp_values):
                return (True, 'partial')
            else:
                gen_strs = extract_string_entities(gen_records)
                exp_strs = extract_string_entities(exp_records)
                if gen_strs and exp_strs:
                    if gen_strs.issubset(exp_strs) or exp_strs.issubset(gen_strs):
                        return (True, 'partial')
                return (False, 'mismatch')
        except Exception as e:
            print(f"⚠️  Error comparing results: {e}")
            return (False, 'error')

    def _is_small_subset_match(self, gen_records: List[Dict], exp_records: List[Dict],
                                threshold: int = 10, overlap_ratio: float = 0.6) -> bool:
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
                if exp_vals and len(gen_vals & exp_vals) / len(exp_vals) >= overlap_ratio:
                    return True
            return False

        for gen_rec in gen_records:
            if not matches_any(record_str_values(gen_rec)):
                return False
        return True

    def evaluate_single_question(self, index: int, item: Dict, use_schema_injection: bool) -> TestResult:
        """Runs the pipeline for a single question and returns metrics."""
        question = item['question']
        expected_cypher = item.get('expected_cypher')
        expected_answer = item.get('ground_truth_answer')

        result = TestResult(
            question_id=index,
            question=question,
            complexity=item.get('complexity', 'unknown'),
            expected_cypher=expected_cypher,
            schema_mode="sliced" if use_schema_injection else "detailed",
        )

        print(f"\n{'='*80}")
        print(f"📝 Q{index}: {question}")
        print(f"   Schema mode: {result.schema_mode.upper()}")
        print(f"{'='*80}")

        # 1. Pipeline (Classification + Translation)
        try:
            # Capture full classifier prediction (active labels, per-label confidence)
            prediction = self.handler.classifier.predict(question)
            result.active_labels = list(prediction.get("labels", []))
            result.per_label_confidence = dict(prediction.get("per_label_confidence", {}))

            # Reset LLM call counter — wrapped _invoke_llm increments it
            self._llm_call_count = 0

            gen_start = time.time()
            handle_result = self.handler.handle(question, use_schema_injection=use_schema_injection)
            result.generation_time_ms = (time.time() - gen_start) * 1000

            result.category = handle_result.category
            result.confidence = handle_result.confidence
            result.schema_name = handle_result.schema_used or ""
            result.generated_cypher = handle_result.cypher or ""
            result.llm_prompt = getattr(self.handler.translator, 'last_initial_prompt', '') or ''
            result.prompt_token_estimate = len(result.llm_prompt) // 4

            # Static schema adherence check (no DB needed)
            if result.generated_cypher:
                adherent, violations = _check_schema_adherence(result.generated_cypher)
                result.schema_adherent = adherent
                result.schema_violations = violations
                if violations:
                    print(f"⚠️  Schema violations detected: {violations}")

            # Capture attempt count (LLM calls during this query)
            result.attempt_count = self._llm_call_count
            result.needed_retry = self._llm_call_count > 1

            # Capture the schema text actually used (truncated for the report)
            # The translator's last_initial_prompt embeds the schema — we keep just a short marker
            if use_schema_injection:
                result.schema_text = f"<slice: {result.schema_name}>"
            else:
                result.schema_text = "<full DETAILED_SCHEMA>"

            if handle_result.error:
                result.error_type = f"Pipeline Error ({handle_result.category})"
                result.error_message = handle_result.error
                result.outcome = "pipeline_error"
                print(f"❌ Pipeline failed: {handle_result.error}")
                return result

            if handle_result.requires_vector_search and not handle_result.cypher:
                result.error_type = "Routing"
                result.error_message = f"Query routed to vector search (category: {handle_result.category})"
                result.outcome = "routing"
                print(f"ℹ️  Routed to Vector Search (skipping Cypher benchmark for this item)")
                return result

        except Exception as e:
            result.error_type = "Execution Exception"
            result.error_message = str(e)
            print(f"❌ Execution error: {e}")
            return result

        # 2. Execution & Accuracy
        # Build runtime parameters for Cypher (vector search needs $queryVector and $k)
        run_params: Dict[str, Any] = {"k": getattr(self.handler, "default_k", 5)}
        if getattr(handle_result, "query_embedding", None) is not None:
            run_params["queryVector"] = handle_result.query_embedding

        try:
            exec_start = time.time()
            with self.neo.driver.session() as session:
                gen_res = list(session.run(result.generated_cypher, **run_params))
                gen_records = [r.data() for r in gen_res]
            result.execution_time_ms = (time.time() - exec_start) * 1000

            with self.neo.driver.session() as session:
                exp_res = list(session.run(expected_cypher, **run_params))
                exp_records = [r.data() for r in exp_res]

            result.generated_results = gen_records
            result.expected_results = exp_records
            result.gen_record_count = len(gen_records)
            result.exp_record_count = len(exp_records)
            result.count_ratio = len(gen_records) / max(len(exp_records), 1)

            print(f"\n🔍 ROUTING: {result.category} ({result.confidence:.2%})")
            print(f"📐 Schema slice: {result.schema_name or '—'}")

            # Full prompt sent to the LLM (initial call)
            if result.llm_prompt:
                print(f"\n{'─'*80}")
                print(f"📤 LLM PROMPT (initial call) — {result.prompt_token_estimate:,} tokens (est.)")
                print("─" * 80)
                print(result.llm_prompt)
                print("─" * 80)

            print(f"Generated Cypher: {result.generated_cypher}")
            print(f"Expected Cypher:  {expected_cypher}")

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

        print(f"⏱️  Timings: Generation={result.generation_time_ms:.0f}ms | Execution={result.execution_time_ms:.0f}ms")
        print(f"🔢 Prompt tokens (est.): {result.prompt_token_estimate:,}")

        return result

    def _run_pipeline(self, test_data: List[Dict], use_schema_injection: bool) -> List[TestResult]:
        """Runs the pipeline once over the entire test set with the given schema mode."""
        mode_label = "SCHEMA SLICING" if use_schema_injection else "DETAILED SCHEMA"
        print(f"\n{'#'*80}")
        print(f"# RUNNING PIPELINE WITH: {mode_label}")
        print(f"{'#'*80}")

        results: List[TestResult] = []
        for i, item in enumerate(test_data, 1):
            res = self.evaluate_single_question(i, item, use_schema_injection=use_schema_injection)
            results.append(res)
        return results

    def run(self, complexity_filter: Optional[List[str]] = None):
        """Main execution loop. Runs the pipelines according to self.mode."""
        test_data = self.load_tests()

        if complexity_filter is None and self.backend == "groq":
            complexity_filter = ["hard"]
            print(f"⚠️  Groq backend detected: Auto-filtering to {complexity_filter} questions only")

        if complexity_filter:
            filter_lower = [f.lower() for f in complexity_filter]
            test_data = [item for item in test_data if item.get('complexity', '').lower() in filter_lower]

        # Report file setup
        reports_dir = self.test_path.parent / "reports"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self.handler.translator.model_name.replace("/", "_").replace(":", "-")
        report_path = reports_dir / f"benchmark_{self.mode}_{timestamp}_{model_slug}.txt"
        benchmark_logger = _BenchmarkLogger(report_path)

        mode_label = {
            "both": "DETAILED vs SLICED",
            "detailed": "DETAILED ONLY",
            "sliced": "SLICED ONLY",
        }.get(self.mode, self.mode)

        print(f"\n{'='*80}")
        print(f"🚀 Text-to-Cypher Benchmark — {mode_label}")
        print(f"{'='*80}")
        print(f"📊 Total Questions: {len(test_data)}")
        print(f"🤖 Code Model: {self.handler.translator.model_name}")
        print(f"🏷️  Classifier: SetFit (multi-label)")
        print(f"📁 Report: {report_path}")
        print(f"{'='*80}\n")

        if self.mode in ("both", "detailed"):
            self.detailed_results = self._run_pipeline(test_data, use_schema_injection=False)

        if self.mode in ("both", "sliced"):
            self.sliced_results = self._run_pipeline(test_data, use_schema_injection=True)

        # Reports
        if self.mode == "both":
            self.print_comparison_report()
        else:
            self.print_single_mode_report()

        # Always print classifier summary + per-question schema/retry table
        self.print_classifier_summary()
        self.print_per_question_table()

        benchmark_logger.close()
        print(f"\n✅ Report saved → {report_path}")

        # Save structured JSON results for llm_judge.py
        json_stem = report_path.stem
        if self.detailed_results:
            self._save_json_results(
                self.detailed_results,
                reports_dir / f"{json_stem}_detailed.json",
            )
        if self.sliced_results:
            self._save_json_results(
                self.sliced_results,
                reports_dir / f"{json_stem}_sliced.json",
            )

        self.cleanup()

    def _compute_stats(self, results: List[TestResult], name: str) -> PipelineStats:
        stats = PipelineStats(name=name, total=len(results))
        if not results:
            return stats

        stats.passed = sum(1 for r in results if r.success)
        stats.failed = stats.total - stats.passed
        stats.accuracy = (stats.passed / stats.total) * 100 if stats.total else 0.0

        gen_times = [r.generation_time_ms for r in results if r.generation_time_ms > 0]
        exec_times = [r.execution_time_ms for r in results if r.execution_time_ms > 0]
        token_estimates = [r.prompt_token_estimate for r in results if r.prompt_token_estimate > 0]

        stats.avg_generation_time_ms = statistics.mean(gen_times) if gen_times else 0.0
        stats.avg_execution_time_ms = statistics.mean(exec_times) if exec_times else 0.0
        stats.avg_prompt_tokens = statistics.mean(token_estimates) if token_estimates else 0.0
        stats.total_prompt_tokens = sum(token_estimates)

        # Retry stats
        attempts = [r.attempt_count for r in results if r.attempt_count > 0]
        stats.retry_count = sum(1 for r in results if r.needed_retry)
        stats.retry_rate = (stats.retry_count / stats.total) * 100 if stats.total else 0.0
        stats.avg_attempts = statistics.mean(attempts) if attempts else 0.0

        # Category breakdown
        for cat in set(r.category for r in results if r.category):
            cat_results = [r for r in results if r.category == cat]
            cat_passed = sum(1 for r in cat_results if r.success)
            stats.category_breakdown[cat] = {
                "total": len(cat_results),
                "passed": cat_passed,
                "accuracy": (cat_passed / len(cat_results)) * 100 if cat_results else 0,
            }

        # Complexity breakdown
        for comp in set(r.complexity for r in results):
            comp_results = [r for r in results if r.complexity == comp]
            comp_passed = sum(1 for r in comp_results if r.success)
            stats.complexity_breakdown[comp] = {
                "total": len(comp_results),
                "passed": comp_passed,
                "accuracy": (comp_passed / len(comp_results)) * 100 if comp_results else 0,
            }

        # ── New metrics ───────────────────────────────────────────────────────
        # First-attempt accuracy (no retries)
        stats.first_attempt_passed = sum(1 for r in results if r.success and r.attempt_count == 1)
        stats.first_attempt_accuracy = (stats.first_attempt_passed / stats.total) * 100 if stats.total else 0.0

        # Outcome breakdown
        stats.outcome_breakdown = dict(Counter(r.outcome for r in results if r.outcome))

        # Count ratio (only for queries that executed — skip routing/pipeline_error)
        ratios = [r.count_ratio for r in results if r.outcome not in ("routing", "pipeline_error", "error", "")]
        stats.avg_count_ratio = statistics.mean(ratios) if ratios else 0.0

        # Schema adherence
        adherent_count = sum(1 for r in results if r.schema_adherent and r.generated_cypher)
        queries_with_cypher = sum(1 for r in results if r.generated_cypher)
        stats.schema_adherence_rate = (adherent_count / queries_with_cypher) * 100 if queries_with_cypher else 0.0
        stats.schema_violations_total = sum(len(r.schema_violations) for r in results)

        # Per-category outcome breakdown
        for cat in set(r.category for r in results if r.category):
            cat_results = [r for r in results if r.category == cat]
            stats.per_category_outcomes[cat] = dict(Counter(r.outcome for r in cat_results if r.outcome))

        return stats

    def print_comparison_report(self):
        """Generates and prints the side-by-side comparison report."""
        det = self._compute_stats(self.detailed_results, "Detailed Schema")
        sli = self._compute_stats(self.sliced_results, "Schema Slicing")

        print("\n" + "=" * 90)
        print("📊 TEXT2CYPHER COMPARISON REPORT — DETAILED vs SLICED")
        print("=" * 90)

        # Summary table
        print(f"\n{'Metric':<35} {'Detailed':<25} {'Sliced':<25}")
        print("-" * 90)
        print(f"{'Total Questions':<35} {det.total:<25} {sli.total:<25}")
        print(f"{'Passed':<35} {det.passed:<25} {sli.passed:<25}")
        print(f"{'Failed':<35} {det.failed:<25} {sli.failed:<25}")
        print(f"{'Accuracy':<35} {det.accuracy:>22.2f} % {sli.accuracy:>22.2f} %")
        print("-" * 90)
        print(f"{'Avg Generation Time (ms)':<35} {det.avg_generation_time_ms:>23.2f}  {sli.avg_generation_time_ms:>23.2f}")
        print(f"{'Avg Execution Time (ms)':<35} {det.avg_execution_time_ms:>23.2f}  {sli.avg_execution_time_ms:>23.2f}")
        print(f"{'Avg Prompt Tokens (est.)':<35} {det.avg_prompt_tokens:>23.0f}  {sli.avg_prompt_tokens:>23.0f}")
        print(f"{'Total Prompt Tokens (est.)':<35} {det.total_prompt_tokens:<25,} {sli.total_prompt_tokens:<25,}")
        print("-" * 90)
        print(f"{'Queries needing retry':<35} {det.retry_count:<25} {sli.retry_count:<25}")
        print(f"{'Retry rate':<35} {det.retry_rate:>22.2f} % {sli.retry_rate:>22.2f} %")
        print(f"{'Avg LLM calls per query':<35} {det.avg_attempts:>23.2f}  {sli.avg_attempts:>23.2f}")
        print("-" * 90)
        print(f"{'First-attempt accuracy':<35} {det.first_attempt_accuracy:>22.2f} % {sli.first_attempt_accuracy:>22.2f} %")
        print(f"{'Avg count ratio (gen/exp)':<35} {det.avg_count_ratio:>23.2f}  {sli.avg_count_ratio:>23.2f}")
        print(f"{'Schema adherence rate':<35} {det.schema_adherence_rate:>22.2f} % {sli.schema_adherence_rate:>22.2f} %")
        print(f"{'Schema violations (total)':<35} {det.schema_violations_total:<25} {sli.schema_violations_total:<25}")

        # Outcome breakdown
        print(f"\n{'─'*90}")
        print("🔎 OUTCOME BREAKDOWN")
        print("─" * 90)
        all_outcomes = sorted(set(list(det.outcome_breakdown.keys()) + list(sli.outcome_breakdown.keys())))
        print(f"{'Outcome':<30} {'Detailed':<25} {'Sliced':<25}")
        print("-" * 90)
        for outcome in all_outcomes:
            d_cnt = det.outcome_breakdown.get(outcome, 0)
            s_cnt = sli.outcome_breakdown.get(outcome, 0)
            print(f"{outcome:<30} {d_cnt:<25} {s_cnt:<25}")

        # Token savings
        if det.total_prompt_tokens > 0:
            savings = (det.total_prompt_tokens - sli.total_prompt_tokens) / det.total_prompt_tokens * 100
            print(f"\n💰 Token Savings (Sliced vs Detailed): {savings:+.1f}%")

        # Speed comparison
        if det.avg_generation_time_ms > 0 and sli.avg_generation_time_ms > 0:
            ratio = det.avg_generation_time_ms / sli.avg_generation_time_ms
            if ratio > 1:
                print(f"⚡ Sliced is {ratio:.2f}x FASTER than Detailed")
            else:
                print(f"⚠️  Sliced is {1/ratio:.2f}x SLOWER than Detailed")

        # Per-complexity breakdown
        print(f"\n{'─'*90}")
        print("📈 ACCURACY BY COMPLEXITY")
        print("─" * 90)
        print(f"{'Complexity':<15} {'Tests':<10} {'Detailed Acc':<20} {'Sliced Acc':<20} {'Δ':<10}")
        print("-" * 90)
        all_comps = set(det.complexity_breakdown.keys()) | set(sli.complexity_breakdown.keys())
        for comp in sorted(all_comps):
            d = det.complexity_breakdown.get(comp, {})
            s = sli.complexity_breakdown.get(comp, {})
            total = d.get("total", 0) or s.get("total", 0)
            d_acc = d.get("accuracy", 0)
            s_acc = s.get("accuracy", 0)
            delta = s_acc - d_acc
            print(f"{comp:<15} {total:<10} {d_acc:>17.1f} % {s_acc:>17.1f} % {delta:>+8.1f}%")

        # Per-category breakdown
        print(f"\n{'─'*90}")
        print("🏷️  ACCURACY BY CATEGORY")
        print("─" * 90)
        print(f"{'Category':<25} {'Tests':<10} {'Detailed Acc':<20} {'Sliced Acc':<20} {'Δ':<10}")
        print("-" * 90)
        all_cats = set(det.category_breakdown.keys()) | set(sli.category_breakdown.keys())
        for cat in sorted(all_cats):
            d = det.category_breakdown.get(cat, {})
            s = sli.category_breakdown.get(cat, {})
            total = d.get("total", 0) or s.get("total", 0)
            d_acc = d.get("accuracy", 0)
            s_acc = s.get("accuracy", 0)
            delta = s_acc - d_acc
            print(f"{cat:<25} {total:<10} {d_acc:>17.1f} % {s_acc:>17.1f} % {delta:>+8.1f}%")

        # Per-category failure mode breakdown (sliced results — most informative)
        if sli.per_category_outcomes:
            print(f"\n{'─'*90}")
            print("🔬 FAILURE MODE BY CATEGORY (SLICED)")
            print("─" * 90)
            outcome_cols = ["pass", "empty_result", "incorrect_result", "error", "pipeline_error", "routing"]
            header = f"{'Category':<25}" + "".join(f"{o:<18}" for o in outcome_cols)
            print(header)
            print("-" * 90)
            for cat in sorted(sli.per_category_outcomes.keys()):
                outcomes = sli.per_category_outcomes[cat]
                row = f"{cat:<25}" + "".join(f"{outcomes.get(o, 0):<18}" for o in outcome_cols)
                print(row)

        # Recommendation
        print(f"\n{'='*90}")
        print("📋 RECOMMENDATION")
        print("=" * 90)
        if sli.accuracy > det.accuracy:
            print(f"✅ Schema Slicing wins on accuracy: {sli.accuracy:.2f}% vs {det.accuracy:.2f}%")
        elif det.accuracy > sli.accuracy:
            print(f"ℹ️  Detailed Schema has higher accuracy: {det.accuracy:.2f}% vs {sli.accuracy:.2f}%")
        else:
            print(f"⚖️  Both achieve identical accuracy: {det.accuracy:.2f}%")

        if sli.avg_prompt_tokens < det.avg_prompt_tokens:
            print(f"💰 Schema Slicing uses fewer tokens — preferred when accuracy is comparable")
        if sli.avg_generation_time_ms < det.avg_generation_time_ms:
            print(f"⚡ Schema Slicing is faster — preferred when accuracy is comparable")

        print("=" * 90)

    # ── Single-mode summary (used when only detailed OR only sliced was run) ──
    def print_single_mode_report(self):
        results = self.sliced_results if self.mode == "sliced" else self.detailed_results
        label = "Schema Slicing" if self.mode == "sliced" else "Detailed Schema"
        s = self._compute_stats(results, label)

        print("\n" + "=" * 90)
        print(f"📊 TEXT2CYPHER REPORT — {label.upper()}")
        print("=" * 90)
        print(f"{'Total Questions':<35} {s.total}")
        print(f"{'Passed':<35} {s.passed}")
        print(f"{'Failed':<35} {s.failed}")
        print(f"{'Accuracy':<35} {s.accuracy:.2f} %")
        print("-" * 90)
        print(f"{'Avg Generation Time (ms)':<35} {s.avg_generation_time_ms:.2f}")
        print(f"{'Avg Execution Time (ms)':<35} {s.avg_execution_time_ms:.2f}")
        print(f"{'Avg Prompt Tokens (est.)':<35} {s.avg_prompt_tokens:.0f}")
        print(f"{'Total Prompt Tokens (est.)':<35} {s.total_prompt_tokens:,}")
        print("-" * 90)
        print(f"{'Queries needing retry':<35} {s.retry_count} / {s.total}")
        print(f"{'Retry rate':<35} {s.retry_rate:.2f} %")
        print(f"{'Avg LLM calls per query':<35} {s.avg_attempts:.2f}")
        print("-" * 90)
        print(f"{'First-attempt accuracy':<35} {s.first_attempt_accuracy:.2f} %")
        print(f"{'Avg count ratio (gen/exp)':<35} {s.avg_count_ratio:.2f}")
        print(f"{'Schema adherence rate':<35} {s.schema_adherence_rate:.2f} %")
        print(f"{'Schema violations (total)':<35} {s.schema_violations_total}")

        # Outcome breakdown
        print(f"\n{'─'*90}")
        print("🔎 OUTCOME BREAKDOWN")
        print("─" * 90)
        for outcome, count in sorted(s.outcome_breakdown.items()):
            pct = count / s.total * 100 if s.total else 0
            print(f"  {outcome:<30} {count:<8} ({pct:.1f}%)")

        # Per-complexity
        print(f"\n{'─'*90}")
        print("📈 ACCURACY BY COMPLEXITY")
        print("─" * 90)
        print(f"{'Complexity':<15} {'Tests':<10} {'Accuracy':<15}")
        print("-" * 90)
        for comp in sorted(s.complexity_breakdown.keys()):
            d = s.complexity_breakdown[comp]
            print(f"{comp:<15} {d['total']:<10} {d['accuracy']:>12.1f} %")

        # Per-category
        print(f"\n{'─'*90}")
        print("🏷️  ACCURACY BY CATEGORY")
        print("─" * 90)
        print(f"{'Category':<25} {'Tests':<10} {'Accuracy':<15}")
        print("-" * 90)
        for cat in sorted(s.category_breakdown.keys()):
            d = s.category_breakdown[cat]
            print(f"{cat:<25} {d['total']:<10} {d['accuracy']:>12.1f} %")

        # Per-category failure modes
        if s.per_category_outcomes:
            print(f"\n{'─'*90}")
            print("🔬 FAILURE MODE BY CATEGORY")
            print("─" * 90)
            outcome_cols = ["pass", "empty_result", "incorrect_result", "error", "pipeline_error", "routing"]
            print(f"{'Category':<25}" + "".join(f"{o:<18}" for o in outcome_cols))
            print("-" * 90)
            for cat in sorted(s.per_category_outcomes.keys()):
                outcomes = s.per_category_outcomes[cat]
                print(f"{cat:<25}" + "".join(f"{outcomes.get(o, 0):<18}" for o in outcome_cols))

    # ── Classifier results summary ────────────────────────────────────────────
    def print_classifier_summary(self):
        """Aggregate classifier output across all questions in the run."""
        # Pick whichever pipeline ran (classifier output is identical for both passes)
        results = self.sliced_results or self.detailed_results
        if not results:
            return

        print("\n" + "=" * 90)
        print("🏷️  QUERY CLASSIFIER RESULTS")
        print("=" * 90)

        # Top-label distribution
        top_label_counts = Counter(r.category for r in results if r.category)
        print(f"\n{'Top label distribution':<35}")
        print("-" * 90)
        print(f"{'Category':<25} {'Count':<10} {'%':<10} {'Avg Confidence':<20}")
        print("-" * 90)
        total = len(results)
        for cat, count in sorted(top_label_counts.items(), key=lambda x: -x[1]):
            cat_results = [r for r in results if r.category == cat]
            avg_conf = statistics.mean([r.confidence for r in cat_results]) if cat_results else 0
            pct = (count / total) * 100
            print(f"{cat:<25} {count:<10} {pct:>7.1f} %  {avg_conf:>15.2%}")

        # Multi-label rate
        multi_label = sum(1 for r in results if len(r.active_labels) > 1)
        single_label = sum(1 for r in results if len(r.active_labels) == 1)
        print("-" * 90)
        print(f"Single-label queries:  {single_label} ({single_label/total*100:.1f}%)")
        print(f"Multi-label queries:   {multi_label} ({multi_label/total*100:.1f}%)")

        # Confidence buckets
        confidences = [r.confidence for r in results if r.confidence > 0]
        if confidences:
            print(f"\nConfidence distribution (top label):")
            print(f"  Avg:    {statistics.mean(confidences):.2%}")
            print(f"  Median: {statistics.median(confidences):.2%}")
            print(f"  Min:    {min(confidences):.2%}")
            print(f"  Max:    {max(confidences):.2%}")
            low_conf = sum(1 for c in confidences if c < 0.5)
            print(f"  Below 50%: {low_conf} ({low_conf/total*100:.1f}%)")

        # Multi-label combinations
        if multi_label:
            print(f"\nMost common multi-label combinations:")
            combo_counts = Counter(
                tuple(sorted(r.active_labels)) for r in results if len(r.active_labels) > 1
            )
            for combo, count in combo_counts.most_common(5):
                print(f"  {' + '.join(combo):<50} {count}")

    # ── Per-question table (schema used + retry needed) ───────────────────────
    def print_per_question_table(self):
        """Print one row per question showing schema slice and retry status."""
        # Use sliced results if available, else detailed
        results = self.sliced_results if self.sliced_results else self.detailed_results
        if not results:
            return

        mode_str = "SLICED" if self.sliced_results else "DETAILED"
        print("\n" + "=" * 110)
        print(f"📋 PER-QUESTION SCHEMA & RETRY TABLE ({mode_str})")
        print("=" * 110)
        print(f"{'#':<5} {'Pass':<5} {'Retry':<7} {'Att':<5} {'Schema':<35} {'Conf':<8} {'Question':<50}")
        print("-" * 110)
        for r in results:
            pass_mark = "✅" if r.success else "❌"
            retry_mark = "🔁" if r.needed_retry else "  "
            schema = (r.schema_name or "—")[:33]
            q_preview = r.question[:48]
            print(f"{r.question_id:<5} {pass_mark:<5} {retry_mark:<7} {r.attempt_count:<5} "
                  f"{schema:<35} {r.confidence:>6.1%}  {q_preview:<50}")

        # Retry summary at the bottom of the table
        retried = [r for r in results if r.needed_retry]
        total = len(results)
        if retried:
            print("-" * 110)
            print(f"🔁 Retry summary: {len(retried)}/{total} ({len(retried)/total*100:.1f}%) queries needed at least one retry")
            print(f"   Of those, {sum(1 for r in retried if r.success)} eventually passed, "
                  f"{sum(1 for r in retried if not r.success)} still failed")
            print(f"   Avg LLM calls per query: {statistics.mean([r.attempt_count for r in results if r.attempt_count > 0]):.2f}")
        print("=" * 110)

    def quick_eval(
        self,
        question: str,
        use_schema_injection: bool = True,
        n: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single question through the Text2Cypher pipeline N times and return
        avg timings + last result. Designed for notebook cells.

        Usage:
            bench = Text2CypherBenchmark(test_set_path=..., model_name=..., backend=...)
            result = bench.quick_eval("What is the expense ratio of VTI?", n=3)
            print(result)

        Returns a dict with:
            question, use_schema_injection, n,
            avg_generation_ms, avg_execution_ms, avg_total_ms,
            avg_attempts, avg_prompt_tokens,
            last_cypher, last_records (up to 5), category, schema_name, success
        """
        item = {"question": question, "expected_cypher": None, "ground_truth_answer": None, "complexity": "unknown"}

        gen_times, exec_times, attempts, tokens = [], [], [], []
        last_result = None

        for i in range(n):
            if verbose and n > 1:
                print(f"\n--- Run {i+1}/{n} ---")
            res = self.evaluate_single_question(index=i + 1, item=item, use_schema_injection=use_schema_injection)
            last_result = res
            if res.generation_time_ms > 0:
                gen_times.append(res.generation_time_ms)
            if res.execution_time_ms > 0:
                exec_times.append(res.execution_time_ms)
            if res.attempt_count > 0:
                attempts.append(res.attempt_count)
            if res.prompt_token_estimate > 0:
                tokens.append(res.prompt_token_estimate)

        avg_gen = statistics.mean(gen_times) if gen_times else 0.0
        avg_exec = statistics.mean(exec_times) if exec_times else 0.0
        avg_total = avg_gen + avg_exec
        avg_att = statistics.mean(attempts) if attempts else 0.0
        avg_tok = statistics.mean(tokens) if tokens else 0.0

        summary = {
            "question": question,
            "use_schema_injection": use_schema_injection,
            "n": n,
            "avg_generation_ms": round(avg_gen, 1),
            "avg_execution_ms": round(avg_exec, 1),
            "avg_total_ms": round(avg_total, 1),
            "avg_attempts": round(avg_att, 2),
            "avg_prompt_tokens": round(avg_tok, 0),
            "category": last_result.category if last_result else "",
            "schema_name": last_result.schema_name if last_result else "",
            "last_cypher": last_result.generated_cypher if last_result else "",
            "last_records": (last_result.generated_results[:5] if last_result else []),
            "success": last_result.success if last_result else False,
            "error": last_result.error_message if last_result and last_result.error_message else None,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUICK EVAL SUMMARY ({n} run{'s' if n > 1 else ''})")
            print(f"{'='*60}")
            print(f"  Schema mode    : {'SLICED' if use_schema_injection else 'DETAILED'}")
            print(f"  Category       : {summary['category']}")
            print(f"  Schema slice   : {summary['schema_name'] or '—'}")
            print(f"  Avg gen time   : {summary['avg_generation_ms']:.1f} ms")
            print(f"  Avg exec time  : {summary['avg_execution_ms']:.1f} ms")
            print(f"  Avg total time : {summary['avg_total_ms']:.1f} ms")
            print(f"  Avg LLM calls  : {summary['avg_attempts']:.2f}")
            print(f"  Avg tokens     : {summary['avg_prompt_tokens']:.0f}")
            print(f"  Last cypher    : {summary['last_cypher']}")
            print(f"  Last records   : {summary['last_records']}")
            print(f"{'='*60}")

        return summary

    def _save_json_results(self, results: List[TestResult], json_path: Path) -> None:
        """Save structured results to JSON for downstream use (e.g. llm_judge.py)."""
        records = []
        for r in results:
            records.append({
                "question_id": r.question_id,
                "question": r.question,
                "complexity": r.complexity,
                "category": r.category,
                "schema_mode": r.schema_mode,
                "schema_name": r.schema_name,
                "confidence": round(r.confidence, 4),
                "active_labels": r.active_labels,
                "outcome": r.outcome,
                "success": r.success,
                "attempt_count": r.attempt_count,
                "needed_retry": r.needed_retry,
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
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON results saved → {json_path}")

    def cleanup(self):
        try:
            self.handler.translator.stop_ollama_server()
        except Exception:
            pass
        try:
            self.neo.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION — used when running with no arguments
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TEST_SET = SCRIPT_DIR / "test_set.json"  # always resolves correctly
DEFAULT_PROVIDER = "openai"        # LM Studio (OpenAI-compatible)
DEFAULT_MODEL = "qwen2.5-coder"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1234


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text2Cypher Schema Comparison Benchmark — runs both pipelines")
    parser.add_argument("--test-set", type=str, default=str(DEFAULT_TEST_SET),
                        help="Path to test set JSON")
    parser.add_argument("--provider", type=str,
                        choices=["ollama", "groq", "openai", "huggingface"],
                        default=DEFAULT_PROVIDER,
                        help="LLM backend (default: openai for LM Studio)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model ID/name")
    parser.add_argument("--openai-host", type=str, default=DEFAULT_HOST,
                        help="OpenAI-compatible host (LM Studio)")
    parser.add_argument("--openai-port", type=int, default=DEFAULT_PORT,
                        help="OpenAI-compatible port (LM Studio default 1234)")
    parser.add_argument("--mode", type=str, choices=["both", "detailed", "sliced"],
                        default="both",
                        help="Which pipeline(s) to run: both | detailed | sliced (default: both)")

    args = parser.parse_args()

    print(f"🔧 Config: provider={args.provider}, model={args.model}, "
          f"host={args.openai_host}, port={args.openai_port}, mode={args.mode}")
    print(f"📂 Test set: {args.test_set}")

    benchmark = Text2CypherBenchmark(
        test_set_path=args.test_set,
        model_name=args.model,
        backend=args.provider,
        interactive=False,
        openai_compatible_host=args.openai_host,
        openai_compatible_port=args.openai_port,
        mode=args.mode,
    )

    try:
        benchmark.run()
    except KeyboardInterrupt:
        print("\n⏸️  Benchmark interrupted by user.")
        benchmark.cleanup()


if __name__ == "__main__":
    main()
