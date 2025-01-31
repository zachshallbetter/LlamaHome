"""Resource monitoring functionality."""

from typing import Dict

import psutil
import torch

from .config import MonitorConfig


class PerformanceMonitor:
    """Monitors system performance metrics."""

    def __init__(self, config: MonitorConfig) -> None:
        """Initialize performance monitor.

        Args:
            config: Monitor configuration
        """
        self.config = config
        self._setup_monitoring()

    def _setup_monitoring(self) -> None:
        """Set up monitoring infrastructure."""
        self.metrics: Dict[str, float] = {}
        self.alerts: Dict[str, bool] = {}

    async def check_resources(self) -> Dict[str, float]:
        """Check current resource utilization.

        Returns:
            Dictionary of resource metrics
        """
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }

        if torch.cuda.is_available():
            metrics.update(
                {
                    f"gpu_{i}_memory": torch.cuda.memory_allocated(i)
                    / torch.cuda.max_memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                }
            )

        return metrics

    async def should_optimize(self) -> bool:
        """Check if optimization is needed.

        Returns:
            True if optimization is needed
        """
        metrics = await self.check_resources()
        return any(
            [
                metrics.get("memory_percent", 0) > self.config.memory_threshold * 100,
                metrics.get("cpu_percent", 0) > self.config.cpu_threshold * 100,
            ]
        )
