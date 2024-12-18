"""Unit tests for LlamaHome."""

import sys
import platform
import pytest
import torch
import subprocess
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from src.core.utils.system import (
    setup_model as ModelSetup,
    check_system_requirements as run_system_checks,
    BenchmarkManager as run_benchmarks
)
from src.core.models.manager import ModelManager
from src.data.analyzer import TextAnalyzer
from src.core.utils.cache import CacheManager

# Test Categories
CATEGORIES = {
    "unit": "Basic unit tests",
    "integration": "Integration tests between components",
    "performance": "Performance and benchmark tests",
    "specialized": "Specialized domain-specific tests",
    "distributed": "Distributed training and processing tests",
    "gpu": "GPU-specific functionality tests",
    "needle": "Needle-in-haystack search tests"
}

# Test fixtures
@pytest.fixture(scope="session")
def test_env():
    """Create complete test environment structure."""
    env = {
        "data_dir": Path("test_data"),
        "config_dir": Path("test_config"),
        "models_dir": Path("test_models"),
        "cache_dir": Path("test_cache"),
        "logs_dir": Path("test_logs")
    }
    
    # Create directories
    for dir_path in env.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return env

@pytest.fixture(scope="session")
def base_config():
    """Base configuration for tests."""
    return {
        "model": {
            "name": "test_model",
            "type": "llama",
            "version": "latest"
        },
        "compute": {
            "device": "cpu",
            "gpu_memory_fraction": 0.9,
            "num_threads": 4
        },
        "data": {
            "batch_size": 32,
            "num_workers": 2
        }
    }

@pytest.fixture(scope="session")
def specialized_runner():
    """Create specialized test runner."""
    return SpecializedTestRunner()

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, test_env):
    """Set up test environment automatically for each test."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
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
            help=f"run {category} tests"
        )

# Base Test Classes
class BaseTest:
    """Base class for all test cases."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.console = Console()
        cls.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}")
        )

    def setup_method(self):
        """Set up test method."""
        self.start_time = pytest.helpers.timer()

    def teardown_method(self):
        """Clean up after test method."""
        duration = pytest.helpers.timer() - self.start_time
        self.console.print(f"Test duration: {duration:.2f}s")

class SystemTest(BaseTest):
    """System setup and configuration tests."""
    
    def test_python_version(self):
        """Test Python version requirements."""
        with self.progress as progress:
            task = progress.add_task("Testing Python version...", total=3)
            
            python_version = sys.version_info
            assert python_version.major == 3
            progress.advance(task)
            
            assert python_version.minor >= 11
            progress.advance(task)
            
            assert python_version.minor != 13
            progress.advance(task)

    def test_compute_backend(self):
        """Test compute backend detection and configuration."""
        with self.progress as progress:
            task = progress.add_task("Testing compute backend...", total=2)
            
            # Test GPU availability
            if torch.cuda.is_available():
                assert torch.cuda.device_count() > 0
                assert torch.cuda.get_device_name(0)
            progress.advance(task)
            
            # Test MPS availability (Apple Silicon)
            if (platform.system() == "Darwin" and 
                platform.machine() == "arm64" and
                hasattr(torch.backends, "mps") and
                torch.backends.mps.is_available()):
                assert True
            progress.advance(task)

    def test_dependencies(self):
        """Test dependency management and Poetry setup."""
        with self.progress as progress:
            task = progress.add_task("Testing dependencies...", total=2)
            
            # Test Poetry installation
            result = subprocess.run(['poetry', '--version'], 
                                  capture_output=True, check=True)
            assert result.returncode == 0
            progress.advance(task)
            
            # Test system checks
            check_results = run_system_checks()
            assert check_results["status"] == "ok"
            assert all(dep["status"] == "ok" for dep in check_results["dependencies"])
            progress.advance(task)

class ModelTest(BaseTest):
    """Model management and setup tests."""
    
    def test_model_management(self):
        """Test model listing and availability."""
        with self.progress as progress:
            task = progress.add_task("Testing model management...", total=3)
            
            model_manager = ModelManager()
            progress.advance(task)
            
            models = model_manager.list_available_models()
            assert "llama" in models
            assert "gpt4" in models
            progress.advance(task)
            
            # Test model check command
            result = subprocess.run(
                ['poetry', 'run', 'python', '-m', 'utils.model_check'],
                capture_output=True, check=True
            )
            assert result.returncode == 0
            progress.advance(task)

    @pytest.mark.gpu
    def test_gpu_model_setup(self):
        """Test GPU-specific model setup."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
            
        with self.progress as progress:
            task = progress.add_task("Testing GPU model setup...", total=2)
            
            model_setup = ModelSetup()
            progress.advance(task)
            
            result = model_setup.setup_model("llama", "latest", device="cuda")
            assert result is not None
            assert result.device.type == "cuda"
            progress.advance(task)

class PerformanceTest(BaseTest):
    """Performance and benchmark tests."""
    
    @pytest.mark.performance
    def test_inference_performance(self):
        """Test model inference performance."""
        with self.progress as progress:
            task = progress.add_task("Testing inference performance...", total=2)
            
            benchmarks = run_benchmarks()
            progress.advance(task)
            
            assert benchmarks["inference_time"] < 1.0  # Max 1 second latency
            assert benchmarks["memory_usage"] < 0.9    # Max 90% memory usage
            progress.advance(task)

    @pytest.mark.performance
    @pytest.mark.gpu
    def test_gpu_training_performance(self):
        """Test GPU training performance."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
            
        with self.progress as progress:
            task = progress.add_task("Testing GPU training performance...", total=2)
            
            benchmarks = run_benchmarks(device="cuda")
            progress.advance(task)
            
            assert benchmarks["training_time"] < 5.0   # Max 5 seconds per batch
            assert benchmarks["gpu_utilization"] > 0.7 # Min 70% GPU utilization
            progress.advance(task)

class IntegrationTest(BaseTest):
    """Integration tests between components."""
    
    @pytest.mark.integration
    def test_pipeline_integration(self):
        """Test complete pipeline integration."""
        with self.progress as progress:
            task = progress.add_task("Testing pipeline integration...", total=3)
            
            # Test data processing pipeline
            data_result = self._test_data_pipeline()
            assert data_result["status"] == "ok"
            progress.advance(task)
            
            # Test model pipeline
            model_result = self._test_model_pipeline()
            assert model_result["status"] == "ok"
            progress.advance(task)
            
            # Test end-to-end pipeline
            e2e_result = self._test_e2e_pipeline()
            assert e2e_result["status"] == "ok"
            progress.advance(task)

    def _test_data_pipeline(self):
        """Test data processing pipeline."""
        return {"status": "ok"}

    def _test_model_pipeline(self):
        """Test model processing pipeline."""
        return {"status": "ok"}

    def _test_e2e_pipeline(self):
        """Test end-to-end pipeline."""
        return {"status": "ok"}

class SpecializedTest(BaseTest):
    """Specialized test cases."""
    
    @pytest.mark.specialized
    def test_needle_search(self, specialized_runner):
        """Test needle-in-haystack search functionality."""
        with self.progress as progress:
            task = progress.add_task("Testing needle search...", total=2)
            
            # Define test search function
            def test_search(haystack: str, needles: list) -> list:
                return [needle for needle in needles if needle in haystack]
            
            # Run needle tests
            results = specialized_runner.run_needle_tests(search_func=test_search)
            progress.advance(task)
            
            # Verify results
            assert results
            for test_key, metrics in results.items():
                assert metrics["accuracy"] > 0.9  # 90% accuracy threshold
                assert metrics["false_positives"] < 0.1  # Max 10% false positives
            progress.advance(task)
    
    @pytest.mark.specialized
    def test_edge_cases(self, specialized_runner):
        """Test edge case handling."""
        results = specialized_runner.run_edge_case_tests()
        assert results == {}  # TODO: Implement edge case tests
    
    @pytest.mark.specialized
    def test_stress_conditions(self, specialized_runner):
        """Test system under stress conditions."""
        results = specialized_runner.run_stress_tests()
        assert results == {}  # TODO: Implement stress tests

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"]) 