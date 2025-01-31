"""Test suite for LlamaHome."""

import unittest
from pathlib import Path
from typing import Dict

from src.core.utils.io import safe_torch_load, safe_torch_save
from src.core.utils.log_manager import LogManager, LogTemplates
from src.core.utils.memory_tracker import MemoryTracker


class TestCore(unittest.TestCase):
    """Test core functionality."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
        self.memory_tracker = MemoryTracker()

    def test_memory_tracking(self) -> None:
        """Test memory tracking functionality."""
        memory_stats = self.memory_tracker.get_memory_stats()
        self.assertIsInstance(memory_stats, dict)

    def test_safe_io(self) -> None:
        """Test safe I/O operations."""
        test_data = {"test": "data"}
        test_path = Path("test_data.pt")

        try:
            # Test save
            hash_value = safe_torch_save(test_data, test_path)
            self.assertIsInstance(hash_value, str)

            # Test load
            loaded_data = safe_torch_load(test_path)
            self.assertEqual(loaded_data, test_data)

        finally:
            # Cleanup
            if test_path.exists():
                test_path.unlink()
            if Path("test_data.hash").exists():
                Path("test_data.hash").unlink()
            if Path("test_data.json").exists():
                Path("test_data.json").unlink()


def run_tests() -> None:
    """Run test suite."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
