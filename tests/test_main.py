"""Unit tests for LlamaHome."""

import os
import sys
import platform
import pytest
import torch
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils.system_check import check_system_requirements
from src.core.setup_env import setup_model
from src.core.utils.benchmark import BenchmarkManager
from src.core.utils.cache_manager import CacheManager
from src.core.utils.io import safe_load_torch
from src.training.data import DataProcessor
from src.core.models.manager import ModelManager

# Test Categories
CATEGORIES = {
    "unit": "Basic unit tests",
    "integration": "Integration tests between components",
    "performance": "Performance and benchmark tests",
    "specialized": "Specialized domain-specific tests",
    "distributed": "Distributed training and processing tests",
    "gpu": "GPU-specific functionality tests",
    "needle": "Needle-in-haystack search tests",
}

class SpecializedTestRunner:
    """Runner for specialized test cases."""

    def __init__(self):
        """Initialize test runner."""
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}")
        )

    def run_needle_tests(self, search_func: callable) -> Dict[str, Dict[str, float]]:
        """Run needle-in-haystack search tests.
        
        Args:
            search_func: Function to test search functionality
            
        Returns:
            Dictionary of test results with metrics
        """
        results = {}
        test_cases = [
            {
                "haystack": "This is a test string with a needle in it",
                "needles": ["needle"],
                "expected": ["needle"]
            },
            {
                "haystack": "Multiple needles in this haystack: needle1, needle2",
                "needles": ["needle1", "needle2", "needle3"],
                "expected": ["needle1", "needle2"]
            }
        ]

        for i, test_case in enumerate(test_cases):
            found = search_func(test_case["haystack"], test_case["needles"])
            expected = test_case["expected"]
            
            # Calculate metrics
            true_positives = len(set(found) & set(expected))
            false_positives = len(set(found) - set(expected))
            false_negatives = len(set(expected) - set(found))
            
            accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
            
            results[f"test_case_{i}"] = {
                "accuracy": accuracy,
                "false_positives": false_positives / len(test_case["needles"]) if test_case["needles"] else 0
            }

        return results

    def run_edge_case_tests(self) -> Dict:
        """Run edge case tests.
        
        Returns:
            Dictionary of test results
        """
        # TODO: Implement edge case tests
        return {}

    def run_stress_tests(self) -> Dict:
        """Run stress tests.
        
        Returns:
            Dictionary of test results
        """
        # TODO: Implement stress tests
        return {}

def test_python_version():
    """Test Python version requirements."""
    python_version = sys.version_info
    assert python_version.major == 3
    assert python_version.minor >= 11
    assert python_version.minor < 13

def test_dependencies():
    """Test dependency installation."""
    try:
        # Test pip installation
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            check=True
        )
        assert result.returncode == 0

        # Test system checks
        check_results = check_system_requirements()
        assert check_results["status"] == "ok"
        assert all(dep["status"] == "ok" for dep in check_results["dependencies"])
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Dependency check failed: {e}")

# Test fixtures
@pytest.fixture(scope="session")
def test_env():
    """Create complete test environment structure."""
    env = {
        "data_dir": Path("test_data"),
        "config_dir": Path("test_config"),
        "models_dir": Path("test_models"),
        "cache_dir": Path("test_cache"),
        "logs_dir": Path("test_logs"),
    }

    # Create directories
    for dir_path in env.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return env

@pytest.fixture(scope="session")
def base_config():
    """Base configuration for tests."""
    return {
        "model": {"name": "test_model", "type": "llama", "version": "latest"},
        "compute": {"device": "cpu", "gpu_memory_fraction": 0.9, "num_threads": 4},
        "data": {"batch_size": 32, "num_workers": 2},
    }

@pytest.fixture(scope="session")
def specialized_runner():
    """Create specialized test runner."""
    return SpecializedTestRunner()

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, test_env):
    """Set up test environment automatically for each test."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Setting up test environment...", total=len(test_env))

        for env_var, path in test_env.items():
            monkeypatch.setenv(f"LLAMAHOME_{env_var.upper()}", str(path))
            progress.advance(task)

    return monkeypatch

def pytest_configure(config):
    """Configure pytest with custom markers and categories."""
    for category, description in CATEGORIES.items():
        config.addinivalue_line("markers", f"{category}: {description}")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on categories and conditions."""
    for category in CATEGORIES:
        if not config.getoption(f"--{category}"):
            skip = pytest.mark.skip(reason=f"need --{category} option to run")
            for item in items:
                if category in item.keywords:
                    item.add_marker(skip)

def pytest_addoption(parser):
    """Add custom command line options for test categories."""
    for category in CATEGORIES:
        parser.addoption(
            f"--{category}",
            action="store_true",
            default=False,
            help=f"run {category} tests",
        )

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])
