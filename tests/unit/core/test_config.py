"""Configuration tests."""

from pathlib import Path
from typing import Dict, Any

import pytest

from src.core.config import ConfigError, ConfigManager
from src.core.metrics.config import MetricsConfig
from src.core.monitoring.config import MonitoringConfig
from src.core.config.base import (
    BaseConfig,
    ResourceConfig,
    ProcessingConfig,
    OptimizationConfig
)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create temporary config directory."""
    return tmp_path / "config"


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration data."""
    return {
        "resources": {
            "gpu_memory_fraction": 0.8,
            "cpu_usage_threshold": 0.7,
            "max_workers": 2,
            "io_queue_size": 500
        },
        "processing": {
            "batch_size": 16,
            "max_sequence_length": 256,
            "num_workers": 2
        },
        "optimization": {
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "warmup_steps": 50
        }
    }


@pytest.fixture
async def config_manager(config_dir: Path) -> ConfigManager:
    """Create config manager instance."""
    config_dir.mkdir(parents=True, exist_ok=True)
    return ConfigManager(config_dir)


async def test_load_from_file(config_dir: Path, test_config: Dict[str, Any]):
    """Test loading configuration from file."""
    # Create test config file
    config_path = config_dir / "test_config.toml"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    import toml
    with open(config_path, "w") as f:
        toml.dump(test_config["resources"], f)
    
    # Load config
    config = ResourceConfig.load_from_file(config_path)
    
    # Verify loaded values
    assert config.gpu_memory_fraction == 0.8
    assert config.cpu_usage_threshold == 0.7
    assert config.max_workers == 2
    assert config.io_queue_size == 500


async def test_load_from_env(monkeypatch: pytest.MonkeyPatch):
    """Test loading configuration from environment."""
    # Set environment variables
    monkeypatch.setenv("LLAMAHOME_GPU_MEMORY_FRACTION", "0.8")
    monkeypatch.setenv("LLAMAHOME_CPU_USAGE_THRESHOLD", "0.7")
    monkeypatch.setenv("LLAMAHOME_MAX_WORKERS", "2")
    monkeypatch.setenv("LLAMAHOME_IO_QUEUE_SIZE", "500")
    
    # Load config
    config = ResourceConfig.load_from_env("LLAMAHOME_")
    
    # Verify loaded values
    assert config.gpu_memory_fraction == 0.8
    assert config.cpu_usage_threshold == 0.7
    assert config.max_workers == 2
    assert config.io_queue_size == 500


async def test_save_to_file(config_dir: Path, test_config: Dict[str, Any]):
    """Test saving configuration to file."""
    # Create config instance
    config = ResourceConfig(**test_config["resources"])
    
    # Save config
    config_path = config_dir / "test_config.toml"
    config_dir.mkdir(parents=True, exist_ok=True)
    config.save_to_file(config_path)
    
    # Verify saved file
    import toml
    loaded = toml.load(config_path)
    assert loaded["gpu_memory_fraction"] == 0.8
    assert loaded["cpu_usage_threshold"] == 0.7
    assert loaded["max_workers"] == 2
    assert loaded["io_queue_size"] == 500


async def test_merge_configs():
    """Test merging configurations."""
    # Create base config
    base = ResourceConfig(
        gpu_memory_fraction=0.8,
        cpu_usage_threshold=0.7,
        max_workers=2,
        io_queue_size=500
    )
    
    # Create override config
    override = ResourceConfig(
        gpu_memory_fraction=0.9,
        cpu_usage_threshold=0.8
    )
    
    # Merge configs
    base.merge(override)
    
    # Verify merged values
    assert base.gpu_memory_fraction == 0.9
    assert base.cpu_usage_threshold == 0.8
    assert base.max_workers == 2
    assert base.io_queue_size == 500


async def test_validation():
    """Test configuration validation."""
    # Test invalid gpu_memory_fraction
    with pytest.raises(ValueError):
        ResourceConfig(
            gpu_memory_fraction=1.5,
            cpu_usage_threshold=0.7,
            max_workers=2,
            io_queue_size=500
        )
    
    # Test invalid max_workers
    with pytest.raises(ValueError):
        ResourceConfig(
            gpu_memory_fraction=0.8,
            cpu_usage_threshold=0.7,
            max_workers=0,
            io_queue_size=500
        )


async def test_config_manager(
    config_manager: ConfigManager,
    test_config: Dict[str, Any]
):
    """Test configuration manager functionality."""
    # Save test configs
    for name, data in test_config.items():
        path = config_manager.config_dir / f"{name}_config.toml"
        import toml
        with open(path, "w") as f:
            toml.dump(data, f)
    
    # Load configs
    resources = await config_manager.load_config(
        ResourceConfig,
        "resources",
        "resources_config.toml"
    )
    processing = await config_manager.load_config(
        ProcessingConfig,
        "processing",
        "processing_config.toml"
    )
    optimization = await config_manager.load_config(
        OptimizationConfig,
        "optimization",
        "optimization_config.toml"
    )
    
    # Verify loaded configs
    assert resources.gpu_memory_fraction == 0.8
    assert processing.batch_size == 16
    assert optimization.learning_rate == 1e-4


async def test_config_updates(config_manager: ConfigManager):
    """Test configuration updates."""
    # Load initial config
    config = await config_manager.load_config(
        ResourceConfig,
        "resources",
        "resources_config.toml"
    )
    
    # Update config
    updates = {"gpu_memory_fraction": 0.9}
    await config_manager.update_config("resources", updates)
    
    # Verify updates
    updated = await config_manager.get_config("resources")
    assert updated.gpu_memory_fraction == 0.9


def test_config_manager_singleton():
    """Test ConfigManager singleton pattern."""
    cm1 = ConfigManager()
    cm2 = ConfigManager()
    assert cm1 is cm2


def test_path_config():
    """Test path configuration."""
    cm = ConfigManager()
    assert cm.paths.get("models").exists()
    assert cm.paths.get("metrics").exists()
    assert cm.paths.get("logs").exists()


def test_toml_loading():
    """Test TOML configuration loading."""
    cm = ConfigManager()
    assert "model" in cm.configs
    assert "training" in cm.configs
    assert "distributed" in cm.configs


def test_monitoring_config():
    """Test monitoring configuration."""
    mc = MonitoringConfig()
    assert mc.log_interval > 0
    assert isinstance(mc.tensorboard, bool)
    assert mc.metrics_dir.exists()


def test_metrics_config():
    """Test metrics configuration."""
    mc = MetricsConfig()
    assert len(mc.enabled_metrics) > 0
    assert mc.aggregation_interval > 0
    assert mc.export_dir.exists()


def test_invalid_config():
    """Test configuration validation."""
    with pytest.raises(ConfigError):
        ConfigManager()._load_toml(Path("nonexistent.toml"))


def test_env_override():
    """Test environment variable override."""
    import os

    os.environ["LLAMAHOME_LOG_INTERVAL"] = "42"
    mc = MonitoringConfig()
    assert mc.log_interval == 42
