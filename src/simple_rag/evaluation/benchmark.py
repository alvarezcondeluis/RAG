import sys
import time
import json
import statistics
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


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
        llama_cpp_host: str = "localhost",
        llama_cpp_port: int = 8080,
    ):
        self.test_path = Path(test_set_path)
        self.neo = Neo4jDatabase()

        # Build extra kwargs for CypherTranslator (forwarded via QueryHandler **cypher_kwargs)
        extra_kwargs = {}
        if backend == "openai":
            extra_kwargs["llama_cpp_host"] = llama_cpp_host
            extra_kwargs["llama_cpp_port"] = llama_cpp_port

        # Initialize the QueryHandler (Classification → Schema Slice → Cypher)
        self.handler = QueryHandler(
            neo4j_driver=self.neo.driver,
            cypher_model=model_name,
            cypher_backend=backend,
            use_entity_resolver=True,
            **extra_kwargs,
        )

        self.results: List[TestResult] = []
        self.backend = backend
        self.interactive = interactive
        self.incorrect_queries: List[Dict[str, Any]] = []

        
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
                        
                return (False, 'mismatch')
            
        except Exception as e:
            print(f"⚠️  Error comparing results: {e}")
            return (False, 'error')

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
            handle_result = self.handler.handle(question)
            result.generation_time_ms = (time.time() - gen_start) * 1000
            
            result.category = handle_result.category
            result.confidence = handle_result.confidence
            result.generated_cypher = handle_result.cypher or ""

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
            # Run Generated Query
            exec_start = time.time()
            with self.neo.driver.session() as session:
                gen_res = list(session.run(result.generated_cypher))
                gen_records = [r.data() for r in gen_res]
            result.execution_time_ms = (time.time() - exec_start) * 1000
            
            # Run Expected Query
            with self.neo.driver.session() as session:
                exp_res = list(session.run(expected_cypher))
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
            
            if is_match:
                result.success = True
                result.is_semantically_correct = True
                match_label = "Exact" if match_type == 'exact' else "Partial"
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
            
        print(f"⏱️  Timings: Generation={result.generation_time_ms:.0f}ms | Execution={result.execution_time_ms:.0f}ms")
        
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
        print(f"📊 Total Questions: {len(test_data)}")
        print(f"🤖 Code Model: {self.handler.translator.model_name}")
        print(f"🏷️  Classifier: SetFit (9 categories)")
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

        print("\n" + "="*60)
        print("📊 PIPELINE BENCHMARK REPORT")
        print("="*60)
        print(f"Total Questions:      {total}")
        print(f"Passed (Correct Data): {passed}")
        print(f"Failed:               {failed}")
        print(f"ACCURACY:             {accuracy:.2f}%")
        print("-" * 60)
        print(f"Avg Dispatch Time:    {avg_gen:.2f} ms (Classify + Schema Slice + Translate)")
        print(f"Avg Execution Time:   {avg_exec:.2f} ms")
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