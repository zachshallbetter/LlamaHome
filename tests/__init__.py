"""Test suite initialization.

Test files:
- test_text_analyzer.py: Text analysis and quality assessment tests
- test_text_processor.py: Text preprocessing and cleaning tests
- test_utils.py: Utility function tests"""

from pathlib import Path

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
BENCHMARK_DATA = TEST_DATA_DIR / "benchmark_data.jsonl.gz"
NEEDLE_TEST_DATA = TEST_DATA_DIR / "needle_test_data.jsonl.gz"
TEST_DOCUMENTS = TEST_DATA_DIR / "test_documents"
SAMPLE_DATA = TEST_DATA_DIR / "sample_data.jsonl"

# Test configuration
TEST_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10
}