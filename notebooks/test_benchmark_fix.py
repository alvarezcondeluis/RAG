"""
Quick test to verify the benchmark fix for dictionary comparison error.
"""
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Test the _normalize_records method
from simple_rag.evaluation.benchmark import Text2CypherBenchmark

# Create a dummy benchmark instance
test_path = Path("../src/simple_rag/evaluation/test_set.json")
benchmark = Text2CypherBenchmark(
    test_set_path=str(test_path),
    model_name="test-model",
    backend="ollama",
    interactive=False
)

# Test with records that contain nested dicts (the problematic case)
test_records = [
    {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
    {"name": "Bob", "details": {"age": 25, "city": "LA"}},
]

print("Testing _normalize_records with nested dictionaries...")
try:
    result = benchmark._normalize_records(test_records)
    print("✅ SUCCESS! _normalize_records works with nested dicts")
    print(f"   Result: {result}")
except TypeError as e:
    print(f"❌ FAILED! Error: {e}")

# Cleanup
benchmark.cleanup()
