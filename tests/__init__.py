"""Test suite initialization.

Test files:
- test_text_analyzer.py: Text analysis and quality assessment tests"""

from pathlib import Path

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
BENCHMARK_DATA = TEST_DATA_DIR / "benchmark_data.jsonl.gz"
NEEDLE_TEST_DATA = TEST_DATA_DIR / "needle_test_data.jsonl.gz"
TEST_DOCUMENTS = TEST_DATA_DIR / "test_documents"
