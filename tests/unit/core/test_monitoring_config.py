"""Tests for monitoring configuration."""

import pytest
from pathlib import Path
from typing import Dict, Any

from src.core.monitoring.config import (
    MonitoringConfig,
    LoggingConfig,
    VisualizationConfig,
    AlertConfig
)

@pytest.fixture
def test_monitoring_config() -> Dict[str, Any]:
    """Test monitoring configuration data."""
    return {
        "logging": {
            "log_interval": 60,
            "save_interval": 600,
            "log_level": "INFO",
            "file_logging": True,
            "console_logging": True,
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "visualization": {
            "tensorboard": True,
            "progress_bars": True,
            "plot_metrics": True,
            "update_interval": 10,
            "tensorboard_dir": "metrics/tensorboard"
        },
        "alerts": {
            "enabled": True,
            "alert_on_error": True,
            "alert_on_completion": True,
            "alert_on_threshold": True,
            "notification_backend": "console",
            "throttle_interval": 300
        },
        "monitoring": {
            "resource_monitoring": True,
            "metrics_history_size": 1000,
            "enable_profiling": False,
            "profiling_interval": 3600
        }
    }

async def test_monitoring_config_load(
    config_dir: Path,
    test_monitoring_config: Dict[str, Any]
):
    """Test loading monitoring configuration."""
    # Create test config file
    config_path = config_dir / "monitoring_config.toml"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    import toml
    with open(config_path, "w") as f:
        toml.dump(test_monitoring_config, f)
    
    # Load config
    config = await MonitoringConfig.load(str(config_dir))
    
    # Verify logging config
    assert config.logging.log_interval == 60
    assert config.logging.save_interval == 600
    assert config.logging.log_level == "INFO"
    assert config.logging.file_logging is True
    
    # Verify visualization config
    assert config.visualization.tensorboard is True
    assert config.visualization.progress_bars is True
    assert config.visualization.update_interval == 10
    
    # Verify alerts config
    assert config.alerts.enabled is True
    assert config.alerts.alert_on_error is True
    assert config.alerts.notification_backend == "console"
    
    # Verify monitoring config
    assert config.resource_monitoring is True
    assert config.metrics_history_size == 1000
    assert config.enable_profiling is False

async def test_monitoring_config_validation():
    """Test monitoring configuration validation."""
    # Test invalid intervals
    with pytest.raises(ValueError):
        LoggingConfig(
            log_interval=0,
            save_interval=600,
            log_level="INFO"
        )
    
    with pytest.raises(ValueError):
        VisualizationConfig(
            tensorboard=True,
            progress_bars=True,
            plot_metrics=True,
            update_interval=0
        )
    
    # Test invalid log level
    with pytest.raises(ValueError):
        LoggingConfig(
            log_interval=60,
            save_interval=600,
            log_level="INVALID"
        )
    
    # Test invalid notification backend
    with pytest.raises(ValueError):
        AlertConfig(
            enabled=True,
            alert_on_error=True,
            alert_on_completion=True,
            alert_on_threshold=True,
            notification_backend="invalid"
        ) 