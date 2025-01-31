"""Monitoring configuration."""

from pathlib import Path
from pydantic import Field

from ..config.base import BaseConfig

class LoggingConfig(BaseConfig):
    """Logging configuration."""
    log_interval: int = Field(60, ge=1)  # seconds
    save_interval: int = Field(600, ge=1)  # seconds
    log_level: str = "INFO"
    file_logging: bool = True
    console_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class VisualizationConfig(BaseConfig):
    """Visualization configuration."""
    tensorboard: bool = True
    progress_bars: bool = True
    plot_metrics: bool = True
    update_interval: int = Field(10, ge=1)  # seconds
    tensorboard_dir: Path = Path("metrics/tensorboard")

class AlertConfig(BaseConfig):
    """Alert configuration."""
    enabled: bool = True
    alert_on_error: bool = True
    alert_on_completion: bool = True
    alert_on_threshold: bool = True
    notification_backend: str = "console"  # console, email, slack, etc.
    throttle_interval: int = Field(300, ge=1)  # seconds

class MonitoringConfig(BaseConfig):
    """Monitoring configuration."""
    logging: LoggingConfig
    visualization: VisualizationConfig
    alerts: AlertConfig
    resource_monitoring: bool = True
    metrics_history_size: int = Field(1000, ge=1)
    enable_profiling: bool = False
    profiling_interval: int = Field(3600, ge=1)  # seconds
    
    @classmethod
    async def load(cls, config_dir: str = "config") -> 'MonitoringConfig':
        """Load monitoring configuration."""
        from ..config.manager import ConfigManager
        
        manager = ConfigManager(config_dir)
        return await manager.load_config(
            cls,
            "monitoring",
            "monitoring_config.toml"
        )
