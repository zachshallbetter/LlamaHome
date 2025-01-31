"""Tests for metrics configuration."""

import pytest
from pathlib import Path
from typing import Dict, Any

from src.core.metrics.config import (
    MetricsConfig,
    StorageConfig
)

@pytest.fixture
def test_metrics_config() -> Dict[str, Any]:
    """Test metrics configuration data."""
    return {
        "storage": {
            "storage_type": "local",
            "retention_days": 30,
            "compression": True,
            "export_format": "parquet",
            "metrics_dir": "metrics",
            "export_dir": "metrics/exports"
        },
        "metrics": {
            "enabled_metrics": ["cpu", "memory", "gpu", "throughput"],
            "aggregation_interval": 60,
            "collect_system_metrics": True,
            "collect_model_metrics": True,
            "collect_training_metrics": True
        }
    }

async def test_metrics_config_load(
    config_dir: Path,
    test_metrics_config: Dict[str, Any]
):
    """Test loading metrics configuration."""
    # Create test config file
    config_path = config_dir / "metrics_config.toml"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    import toml
    with open(config_path, "w") as f:
        toml.dump(test_metrics_config, f)
    
    # Load config
    config = await MetricsConfig.load(str(config_dir))
    
    # Verify storage config
    assert config.storage.storage_type == "local"
    assert config.storage.retention_days == 30
    assert config.storage.compression is True
    assert config.storage.export_format == "parquet"
    
    # Verify metrics config
    assert "cpu" in config.enabled_metrics
    assert "memory" in config.enabled_metrics
    assert config.aggregation_interval == 60
    assert config.collect_system_metrics is True

async def test_metrics_config_validation():
    """Test metrics configuration validation."""
    # Test invalid retention days
    with pytest.raises(ValueError):
        StorageConfig(
            storage_type="local",
            retention_days=0,
            compression=True,
            export_format="parquet"
        )
    
    # Test invalid storage type
    with pytest.raises(ValueError):
        StorageConfig(
            storage_type="invalid",
            retention_days=30,
            compression=True,
            export_format="parquet"
        )
    
    # Test invalid export format
    with pytest.raises(ValueError):
        StorageConfig(
            storage_type="local",
            retention_days=30,
            compression=True,
            export_format="invalid"
        ) 