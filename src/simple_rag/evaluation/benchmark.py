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
    
    def __init__(self, test_set_path: str, model_name: str):
        self.test_path = Path(test_set_path)
        self.translator = CypherTranslator(model_name=model_name)
        self.neo = Neo4jDatabase()
        self.results: List[TestResult] = []
        
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
    
    def _compare_results_flexible(self, gen_records: List[Dict], exp_records: List[Dict]) -> bool:
        """
        Compare results flexibly - ignores alias names, only compares values.
        Handles cases where generated query uses different aliases than expected.
        """
        if len(gen_records) != len(exp_records):
            return False
        
        # Extract and sort values only (ignore keys/aliases)
        def extract_values(records):
            result = []
            for record in records:
                # Get all values, convert to strings, and sort them
                values = tuple(sorted([str(v) for v in record.values()]))
                result.append(values)
            return sorted(result)
        
        gen_values = extract_values(gen_records)
        exp_values = extract_values(exp_records)
        
        return gen_values == exp_values

    def evaluate_single_question(self, index: int, item: Dict) -> TestResult:
        """Runs the pipeline for a single question and returns metrics."""
        question = item['question']
        expected_cypher = item.get('expected_cypher')
        
        result = TestResult(
            question_id=index,
            question=question,
            complexity=item.get('complexity', 'unknown'),
            expected_cypher=expected_cypher
        )

        print(f"🔹 Processing Q{index}: {question[:50]}...")

        # 1. Measure Generation
        try:
            gen_start = time.time()
            generated_cypher = self.translator.translate(question)
            result.generation_time_ms = (time.time() - gen_start) * 1000
            result.generated_cypher = generated_cypher

            if not generated_cypher:
                result.error_type = "Generation Failure"
                result.error_message = "Model returned empty string"
                return result

        except Exception as e:
            result.error_type = "Translation Exception"
            result.error_message = str(e)
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
            
            # D. Compare Results (flexible - ignores alias names)
            if self._compare_results_flexible(gen_records, exp_records):
                result.success = True
                result.is_semantically_correct = True
            else:
                result.success = False
                result.error_type = "Incorrect Result"
                result.error_message = f"Got {len(gen_records)} records, expected {len(exp_records)}"
                
        except Exception as e:
            result.success = False
            result.error_type = "Syntax/Execution Error"
            result.error_message = str(e)

        return result

    def run(self):
        """Main execution loop."""
        test_data = self.load_tests()
        print(f"\n🚀 Starting Benchmark on {len(test_data)} questions...\n")
        
        for i, item in enumerate(test_data, 1):
            res = self.evaluate_single_question(i, item)
            self.results.append(res)
            
            # Real-time feedback
            status = "✅ PASS" if res.success else f"❌ FAIL ({res.error_type})"
            print(f"   -> {status} | Gen: {res.generation_time_ms:.0f}ms | Exec: {res.execution_time_ms:.0f}ms")

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

    def cleanup(self):
        self.translator.stop_ollama_server()
        self.neo.close()