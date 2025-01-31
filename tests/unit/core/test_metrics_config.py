"""Test metrics configuration."""

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.core.metrics.config import MetricsConfig


@pytest.fixture
def metrics_config() -> MetricsConfig:
    """Create test metrics configuration.
    
    Returns:
        Test metrics configuration
    """
    return MetricsConfig(
        enabled_metrics=["accuracy", "loss", "f1"],
        aggregation_interval=60,
        export_dir=Path("metrics"),
        prometheus_port=9090,
        tensorboard_enabled=True,
        log_level="INFO",
    )


@pytest.fixture
def metrics_data() -> Dict[str, Any]:
    """Create test metrics data.
    
    Returns:
        Test metrics data
    """
    return {
        "enabled_metrics": ["accuracy", "loss", "f1"],
        "aggregation_interval": 60,
        "export_dir": "metrics",
        "prometheus_port": 9090,
        "tensorboard_enabled": True,
        "log_level": "INFO",
    }


async def test_metrics_config_load(metrics_data: Dict[str, Any]) -> None:
    """Test loading metrics configuration.
    
    Args:
        metrics_data: Test metrics data
    """
    config = MetricsConfig(**metrics_data)
    
    # Verify loaded values
    assert config.enabled_metrics == metrics_data["enabled_metrics"]
    assert config.aggregation_interval == metrics_data["aggregation_interval"]
    assert config.export_dir == Path(metrics_data["export_dir"])
    assert config.prometheus_port == metrics_data["prometheus_port"]
    assert config.tensorboard_enabled == metrics_data["tensorboard_enabled"]
    assert config.log_level == metrics_data["log_level"]


async def test_metrics_config_validation() -> None:
    """Test metrics configuration validation."""
    # Test invalid metrics
    with pytest.raises(ValidationError):
        MetricsConfig(
            enabled_metrics=["invalid_metric"],
            aggregation_interval=60,
            export_dir="metrics",
        )

    # Test invalid interval
    with pytest.raises(ValidationError):
        MetricsConfig(
            enabled_metrics=["accuracy"],
            aggregation_interval=-1,
            export_dir="metrics",
        )

    # Test invalid export directory
    with pytest.raises(ValidationError):
        MetricsConfig(
            enabled_metrics=["accuracy"],
            aggregation_interval=60,
            export_dir="",
        )

    # Test invalid prometheus port
    with pytest.raises(ValidationError):
        MetricsConfig(
            enabled_metrics=["accuracy"],
            aggregation_interval=60,
            export_dir="metrics",
            prometheus_port=70000,
        )

    # Test invalid log level
    with pytest.raises(ValidationError):
        MetricsConfig(
            enabled_metrics=["accuracy"],
            aggregation_interval=60,
            export_dir="metrics",
            log_level="INVALID",
        )
