"""Unit tests for LlamaHome."""

import os
import sys
import platform
import pytest
import torch
import subprocess
import toml
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from utils.log_manager import LogManager, LogTemplates
from src.managers.model_manager import ModelManager
from utils.setup_model import ModelSetup
from utils.cache_manager import CacheManager
from utils.system_check import run_system_checks
from src.data.analyzer import TextAnalyzer
from utils.benchmark import run_benchmarks
from src.testing.needle_test import run_needle_tests

def __init__():
    """Initialize test module."""
    pytest.main()

# Test fixtures
@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def test_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def test_models_dir(tmp_path):
    """Create temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir

@pytest.fixture
def test_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir

@pytest.fixture
def base_model_config():
    """Default test model configuration."""
    return {
        "name": "test_model",
        "requires_gpu": False,
        "compute_backend": "cpu",
        "model_type": "llama",
        "version": "latest",
        "min_gpu_memory": {
            "7b": 12,
            "13b": 24,
            "70b": 100
        },
        "h2o_config": {
            "enable": True,
            "window_length": 1024
        }
    }

@pytest.fixture
def clean_env(monkeypatch):
    """Clean test environment."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MPS_AVAILABLE", raising=False)
    return monkeypatch

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, test_data_dir, test_config_dir, test_models_dir, test_cache_dir):
    """Set up test environment automatically for each test."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Setting up test environment...", total=4)
        
        monkeypatch.setenv("LLAMAHOME_DATA_DIR", str(test_data_dir))
        progress.advance(task)
        
        monkeypatch.setenv("LLAMAHOME_CONFIG_DIR", str(test_config_dir))
        progress.advance(task)
        
        monkeypatch.setenv("LLAMAHOME_MODELS_DIR", str(test_models_dir))
        progress.advance(task)
        
        monkeypatch.setenv("LLAMAHOME_CACHE_DIR", str(test_cache_dir))
        progress.advance(task)
        
    return monkeypatch

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "benchmark: mark as performance benchmark")
    config.addinivalue_line("markers", "needle: mark as needle-in-haystack test")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and conditions."""
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    skip_needle = pytest.mark.skip(reason="need --needle option to run")
    
    if not config.getoption("--runslow"):
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
                
    if not config.getoption("--gpu"):
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    if not config.getoption("--integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if not config.getoption("--benchmark"):
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)

    if not config.getoption("--needle"):
        for item in items:
            if "needle" in item.keywords:
                item.add_marker(skip_needle)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run GPU tests"
    )
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run performance benchmarks"
    )
    parser.addoption(
        "--needle", action="store_true", default=False, help="run needle-in-haystack tests"
    )

# Core tests
class TestSystemSetup:
    """Test system setup and configuration."""
    
    def test_python_version(self):
        with Progress() as progress:
            task = progress.add_task("Testing Python version...", total=3)
            
            python_version = sys.version_info
            assert python_version.major == 3
            progress.advance(task)
            
            assert python_version.minor >= 11
            progress.advance(task)
            
            assert python_version.minor != 13
            progress.advance(task)
        
    def test_compute_backend_detection(self):
        with Progress() as progress:
            task = progress.add_task("Testing compute backend...", total=1)
            
            if torch.cuda.is_available():
                assert torch.cuda.device_count() > 0
                assert torch.cuda.get_device_name(0)
            elif (platform.system() == "Darwin" and 
                  platform.machine() == "arm64" and
                  hasattr(torch.backends, "mps") and
                  torch.backends.mps.is_available()):
                assert True  # MPS available
            else:
                assert platform.processor() or platform.machine()
                
            progress.advance(task)

    def test_poetry_setup(self):
        with Progress() as progress:
            task = progress.add_task("Testing Poetry setup...", total=1)
            
            try:
                result = subprocess.run(['poetry', '--version'], 
                                      capture_output=True, check=True)
                assert result.returncode == 0
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytest.fail("Poetry not installed")
                
            progress.advance(task)

    def test_system_checks(self):
        """Test system check functionality."""
        with Progress() as progress:
            task = progress.add_task("Running system checks...", total=3)
            
            check_results = run_system_checks()
            progress.advance(task)
            
            assert check_results["status"] == "ok"
            progress.advance(task)
            
            assert "dependencies" in check_results
            assert "system" in check_results
            progress.advance(task)

class TestModelManager:
    """Test model management functionality."""
    
    def test_model_listing(self):
        with Progress() as progress:
            task = progress.add_task("Testing model listing...", total=4)
            
            model_manager = ModelManager()
            progress.advance(task)
            
            available_models = model_manager.list_available_models()
            progress.advance(task)
            
            assert "llama" in available_models
            assert "gpt4" in available_models
            assert "claude" in available_models
            progress.advance(task)
            
            # Run model check command
            result = subprocess.run(['poetry', 'run', 'python', '-m', 'utils.model_check'],
                                  capture_output=True, check=True)
            assert result.returncode == 0
            progress.advance(task)
        
    @pytest.mark.slow
    def test_model_setup(self):
        with Progress() as progress:
            task = progress.add_task("Testing model setup...", total=2)
            
            model_setup = ModelSetup()
            progress.advance(task)
            
            result = model_setup.setup_model("llama", "latest")
            assert result is not None
            progress.advance(task)

    @pytest.mark.gpu
    def test_gpu_model_setup(self):
        """Test GPU model setup if available."""
        with Progress() as progress:
            task = progress.add_task("Testing GPU model setup...", total=2)
            
            if not torch.cuda.is_available():
                pytest.skip("No GPU available")
            progress.advance(task)
            
            model_setup = ModelSetup()
            result = model_setup.setup_model("llama", "latest", device="cuda")
            assert result is not None
            progress.advance(task)

class TestLogging:
    """Test logging functionality."""
    
    def test_logger_initialization(self):
        with Progress() as progress:
            task = progress.add_task("Testing logger initialization...", total=3)
            
            logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
            progress.advance(task)
            
            assert logger is not None
            progress.advance(task)
            
            assert logger.name == __name__
            progress.advance(task)
