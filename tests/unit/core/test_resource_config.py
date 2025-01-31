"""Tests for resource configuration."""

from pathlib import Path
from typing import Any, Dict

import pytest
import toml  # type: ignore # missing stubs
import torch

from src.core.resource.config import GPUConfig, MonitorConfig, ResourceConfig


@pytest.fixture
def test_resource_config() -> Dict[str, Any]:
    """Test resource configuration data."""
    return {
        "gpu": {
            "memory_fraction": 0.9,
            "allow_growth": True,
            "per_process_memory": "12GB",
            "enable_tf32": True,
            "cuda_devices": [0, 1],
        },
        "monitor": {
            "check_interval": 1.0,
            "memory_threshold": 0.9,
            "cpu_threshold": 0.8,
            "gpu_temp_threshold": 80.0,
            "alert_on_threshold": True,
            "collect_metrics": True,
        },
        "resource": {"max_workers": 4, "io_queue_size": 1000, "pin_memory": True},
    }


async def test_resource_config_load(
    config_dir: Path, test_resource_config: Dict[str, Any]
) -> None:
    """Test loading resource configuration."""
    # Create test config file
    config_path = config_dir / "resource_config.toml"
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        toml.dump(test_resource_config, f)

    # Load config
    config = await ResourceConfig.load(str(config_dir))

    # Verify GPU config
    assert config.gpu.memory_fraction == 0.9
    assert config.gpu.allow_growth is True
    assert config.gpu.per_process_memory == "12GB"
    assert config.gpu.enable_tf32 is True
    assert config.gpu.cuda_devices == [0, 1]

    # Verify monitor config
    assert config.monitor.check_interval == 1.0
    assert config.monitor.memory_threshold == 0.9
    assert config.monitor.cpu_threshold == 0.8
    assert config.monitor.gpu_temp_threshold == 80.0

    # Verify resource config
    assert config.max_workers == 4
    assert config.io_queue_size == 1000
    assert config.pin_memory is True


async def test_resource_config_validation() -> None:
    """Test resource configuration validation."""
    # Test invalid memory fraction
    with pytest.raises(ValueError):
        GPUConfig(
            memory_fraction=1.5,
            allow_growth=True,
            per_process_memory="12GB",
            enable_tf32=True,
        )

    # Test invalid check interval
    with pytest.raises(ValueError):
        MonitorConfig(
            check_interval=0.0,
            memory_threshold=0.9,
            cpu_threshold=0.8,
            gpu_temp_threshold=80.0,
        )

    # Test invalid thresholds
    with pytest.raises(ValueError):
        MonitorConfig(
            check_interval=1.0,
            memory_threshold=1.5,
            cpu_threshold=0.8,
            gpu_temp_threshold=80.0,
        )


async def test_gpu_memory_detection() -> None:
    """Test GPU memory detection."""
    config = GPUConfig(
        memory_fraction=0.9,
        allow_growth=True,
        per_process_memory="12GB",
        enable_tf32=True,
    )

    if torch.cuda.is_available():
        assert config.available_memory > 0
    else:
        assert config.available_memory == 0
