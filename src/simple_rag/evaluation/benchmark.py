import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Import your existing modules
from simple_rag.rag.text2cypher import CypherTranslator
from simple_rag.database.neo4j.neo4j import Neo4jDatabase

@dataclass
class TestResult:
    """Stores the metrics for a single test case."""
    question_id: int
    question: str
    complexity: str
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
    A professional testing suite for evaluating Text-to-Cypher models.
    """
    
    def __init__(self, test_set_path: str, model_name: str, backend: str, interactive: bool = False):
        self.test_path = Path(test_set_path)
        self.neo = Neo4jDatabase()
        self.translator = CypherTranslator(neo4j_driver=self.neo.driver, model_name=model_name, backend=backend, use_entity_resolver=True)
        
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
        # Convert all values to strings and sort keys to ensure comparable structures
        normalized = []
        for r in records:
            # Flatten or clean dictionary if needed
            normalized.append({k: str(v) for k, v in r.items()})
        # Sort by first key's value to handle order-agnostic comparison
        try:
            return sorted(normalized, key=lambda x: list(x.values())[0])
        except IndexError:
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
                for record in records:
                    if isinstance(record, int):
                        return record
                    for v in record.values():
                        all_values.add(str(v))
                return all_values
            
            gen_values = extract_all_values(gen_records)
            exp_values = extract_all_values(exp_records)
            
            if isinstance(exp_values, int):
                return (str(exp_values) in gen_values, 'exact')
            
            # Check match type
            if gen_values == exp_values:
                return (True, 'exact')
            elif exp_values.issubset(gen_values):
                return (True, 'partial')  # Generated has all expected + more
            elif gen_values.issubset(exp_values):
                return (True, 'partial')  # Generated has subset of expected
            else:
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
        
        # 1. Measure Generation
        try:
            gen_start = time.time()
            generated_cypher = self.translator.translate(question)
            result.generation_time_ms = (time.time() - gen_start) * 1000
            result.generated_cypher = generated_cypher

            if not generated_cypher:
                result.error_type = "Generation Failure"
                result.error_message = "Model returned empty string"
                print(f"❌ Generation failed: Model returned empty string")
                return result

        except Exception as e:
            result.error_type = "Translation Exception"
            result.error_message = str(e)
            print(f"❌ Translation error: {e}")
            return result

        # 2. Measure Execution & Accuracy
        try:
            # A. Run Generated Query
            exec_start = time.time()
            with self.neo.driver.session() as session:
                gen_res = list(session.run(generated_cypher))
                gen_records = [r.data() for r in gen_res]
            result.execution_time_ms = (time.time() - exec_start) * 1000
            
            with self.neo.driver.session() as session:
                exp_res = list(session.run(expected_cypher))
                exp_records = [r.data() for r in exp_res]

            # C. Store results for comparison
            result.generated_results = gen_records
            result.expected_results = exp_records
            
            # D. Display both queries
            print(f"\n🔍 QUERIES:")
            print(f"Generated: {generated_cypher}")
            print(f"Expected:  {expected_cypher}")
            
            # E. Display both results
            print(f"\n📊 RESULTS:")
            print(f"Generated ({len(gen_records)} records):")
            for i, record in enumerate(gen_records[:3]):  # Show first 3 records
                print(f"  [{i+1}] {record}")
            if len(gen_records) > 3:
                print(f"  ... and {len(gen_records)-3} more")
            
            print(f"Expected ({len(exp_records)} records):")
            for i, record in enumerate(exp_records[:3]):  # Show first 3 records
                print(f"  [{i+1}] {record}")
            if len(exp_records) > 3:
                print(f"  ... and {len(exp_records)-3} more")
            
            # F. Compare Results (flexible - ignores alias names)
            is_match, match_type = self._compare_results_flexible(gen_records, exp_records)
            
            if not is_match and expected_answer:
                is_match, match_type = self._compare_results_flexible(gen_records, expected_answer)
            
            if is_match:
                result.success = True
                result.is_semantically_correct = True
                if match_type == 'exact':
                    print(f"\n✅ PASS - Exact match")
                elif match_type == 'partial':
                    print(f"\n🟠 PASS - Partial match (extra or missing fields)")
                    print(f"   Generated: {len(gen_records)} records")
                    print(f"   Expected: {len(exp_records)} records")
            else:
                result.success = False
                result.error_type = "Incorrect Result"
                result.error_message = f"Got {len(gen_records)} records, expected {len(exp_records)}"
                print(f"\n❌ FAIL - Result mismatch")
                print(f"   Generated: {gen_records[:2] if len(gen_records) > 0 else 'empty'}...")
                print(f"   Expected: {exp_records[:2] if len(exp_records) > 0 else 'empty'}...")
                
        except Exception as e:
            result.success = False
            result.error_type = "Syntax/Execution Error"
            result.error_message = str(e)
            print(f"\n❌ FAIL - Execution error: {str(e)[:100]}")
            
        print(f"\n⏱️  Timings: Generation={result.generation_time_ms:.0f}ms | Execution={result.execution_time_ms:.0f}ms")
        
        # Interactive validation if enabled
        if self.interactive:
            user_validation = self._ask_user_validation(result, item)
            if not user_validation:
                self._save_incorrect_query(result, item)
        
        return result

    def run(self, complexity_filter: Optional[List[str]] = None):
        """
        Main execution loop.
        
        Args:
            complexity_filter: List of complexity levels to include ['simple', 'medium', 'hard'].
                              If None and backend is 'groq', defaults to ['hard'] to conserve API limits.
                              Examples: ['hard'], ['medium', 'hard'], ['simple', 'medium', 'hard']
        """
        test_data = self.load_tests()
        
        # Auto-filter for Groq backend to conserve API limits
        if complexity_filter is None and self.backend == "groq":
            complexity_filter = ["hard"]
            print(f"⚠️  Groq backend detected: Auto-filtering to {complexity_filter} questions only to conserve API limits")
        
        # Apply complexity filter if specified
        if complexity_filter:
            original_count = len(test_data)
            # Normalize filter values to lowercase for comparison
            filter_lower = [f.lower() for f in complexity_filter]
            test_data = [item for item in test_data if item.get('complexity', '').lower() in filter_lower]
            print(f"🔍 Filtered: {original_count} → {len(test_data)} questions (complexity={complexity_filter})")
        
        print(f"\n{'='*80}")
        print(f"🚀 Text-to-Cypher Benchmark Suite")
        print(f"{'='*80}")
        print(f"📊 Total Questions: {len(test_data)}")
        print(f"🤖 Model: {self.translator.model_name}")
        print(f"🔧 Backend: {self.backend}")
        if complexity_filter:
            print(f"🎯 Complexity Filter: {', '.join(complexity_filter)}")
        print(f"{'='*80}\n")
        
        for i, item in enumerate(test_data, 1):
            res = self.evaluate_single_question(i, item)
            self.results.append(res)

        self.print_report()
        self.cleanup()

    def print_report(self):
        """Generates and prints the final metrics report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        accuracy = (passed / total) * 100 if total > 0 else 0
        
        # Calculate Latencies (only for successful/attempted ops)
        gen_times = [r.generation_time_ms for r in self.results if r.generation_time_ms > 0]
        exec_times = [r.execution_time_ms for r in self.results if r.execution_time_ms > 0]
        
        avg_gen = statistics.mean(gen_times) if gen_times else 0
        avg_exec = statistics.mean(exec_times) if exec_times else 0

        print("\n" + "="*60)
        print("📊 TEXT-TO-CYPHER BENCHMARK REPORT")
        print("="*60)
        print(f"Total Questions:      {total}")
        print(f"Passed (Correct Data): {passed}")
        print(f"Failed:               {failed}")
        print(f"ACCURACY:             {accuracy:.2f}%")
        print("-" * 60)
        print(f"Avg Generation Time:  {avg_gen:.2f} ms")
        print(f"Avg Execution Time:   {avg_exec:.2f} ms")
        print("="*60)
        
        if failed > 0:
            print("\n🚩 ERROR LOG (Failed Questions):")
            for r in self.results:
                if not r.success:
                    print(f"\n{'='*80}")
                    print(f"❌ Q{r.question_id} [{r.complexity.upper()}]: {r.question}")
                    print(f"{'='*80}")
                    print(f"\n📋 Error Type: {r.error_type}")
                    print(f"💬 Message: {r.error_message}")
                    
                    print(f"\n🔹 EXPECTED CYPHER:")
                    print(f"   {r.expected_cypher}")
                    
                    print(f"\n🔸 GENERATED CYPHER:")
                    print(f"   {r.generated_cypher}")
                    
                    if r.expected_results or r.generated_results:
                        print(f"\n📊 RESULTS COMPARISON:")
                        print(f"\n   ✅ Expected Results ({len(r.expected_results)} records):")
                        if r.expected_results:
                            for i, rec in enumerate(r.expected_results[:5], 1):  # Show first 5
                                print(f"      {i}. {rec}")
                            if len(r.expected_results) > 5:
                                print(f"      ... and {len(r.expected_results) - 5} more")
                        else:
                            print("      (empty result set)")
                        
                        print(f"\n   ❌ Generated Results ({len(r.generated_results)} records):")
                        if r.generated_results:
                            for i, rec in enumerate(r.generated_results[:5], 1):  # Show first 5
                                print(f"      {i}. {rec}")
                            if len(r.generated_results) > 5:
                                print(f"      ... and {len(r.generated_results) - 5} more")
                        else:
                            print("      (empty result set)")
                    
                    print(f"\n{'='*80}\n")

    def _ask_user_validation(self, result: TestResult, item: Dict) -> bool:
        """
        Ask user if the generated query and results are correct.
        Returns True if correct, False if incorrect.
        """
        print("\n" + "="*80)
        print("🔍 USER VALIDATION REQUIRED")
        print("="*80)
        
        while True:
            response = input("\n❓ Is this query and result CORRECT? (y/n/s to skip): ").strip().lower()
            
            if response == 'y':
                print("✅ Marked as CORRECT")
                return True
            elif response == 'n':
                print("❌ Marked as INCORRECT - will be saved for review")
                # Ask for optional feedback
                feedback = input("💬 Optional feedback (press Enter to skip): ").strip()
                if feedback:
                    result.error_message = f"User feedback: {feedback}"
                return False
            elif response == 's':
                print("⏭️  Skipped - treating as correct")
                return True
            else:
                print("⚠️  Invalid input. Please enter 'y' (yes), 'n' (no), or 's' (skip)")
    
    def _save_incorrect_query(self, result: TestResult, item: Dict):
        """
        Save an incorrect query to the list for later export.
        """
        incorrect_entry = {
            "question_id": result.question_id,
            "question": result.question,
            "complexity": result.complexity,
            "expected_cypher": result.expected_cypher,
            "generated_cypher": result.generated_cypher,
            "expected_results": result.expected_results,
            "generated_results": result.generated_results,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "generation_time_ms": result.generation_time_ms,
            "execution_time_ms": result.execution_time_ms
        }
        self.incorrect_queries.append(incorrect_entry)
        print(f"💾 Saved to incorrect queries list (total: {len(self.incorrect_queries)})")
    
    def export_incorrect_queries(self, output_path: str = None):
        """
        Export incorrect queries to a JSON file.
        """
        if not self.incorrect_queries:
            print("\n✅ No incorrect queries to export!")
            return
        
        if output_path is None:
            # Default to timestamp-based filename in test_results folder
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create test_results folder if it doesn't exist
            results_dir = Path(__file__).parent / "test_results"
            results_dir.mkdir(exist_ok=True)
            
            output_path = results_dir / f"incorrect_queries_{timestamp}.json"
        
        output_file = Path(output_path)
        
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "total_incorrect": len(self.incorrect_queries),
                    "model": self.translator.model_name,
                    "backend": self.backend
                },
                "incorrect_queries": self.incorrect_queries
            }, f, indent=2)
        
        print(f"\n💾 Exported {len(self.incorrect_queries)} incorrect queries to: {output_file}")
        print(f"📁 Full path: {output_file.absolute()}")
    
    def cleanup(self):
        # Export incorrect queries if any were collected
        if self.interactive and self.incorrect_queries:
            self.export_incorrect_queries()
        
        self.translator.stop_ollama_server()
        self.neo.close()