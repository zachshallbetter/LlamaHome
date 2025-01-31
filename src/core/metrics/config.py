"""Metrics configuration."""

from typing import Set
from pathlib import Path
from pydantic import Field

from ..config.base import BaseConfig

class StorageConfig(BaseConfig):
    """Metrics storage configuration."""
    storage_type: str = "local"  # local, s3, etc.
    retention_days: int = Field(30, ge=1)
    compression: bool = True
    export_format: str = "parquet"
    metrics_dir: Path = Path("metrics")
    export_dir: Path = Path("metrics/exports")

class MetricsConfig(BaseConfig):
    """Metrics configuration."""
    enabled_metrics: Set[str] = {"cpu", "memory", "gpu", "throughput"}
    aggregation_interval: int = Field(60, ge=1)  # seconds
    storage: StorageConfig
    collect_system_metrics: bool = True
    collect_model_metrics: bool = True
    collect_training_metrics: bool = True
    
    @classmethod
    async def load(cls, config_dir: str = "config") -> 'MetricsConfig':
        """Load metrics configuration."""
        from ..config.manager import ConfigManager
        
        manager = ConfigManager(config_dir)
        return await manager.load_config(
            cls,
            "metrics",
            "metrics_config.toml"
        )
