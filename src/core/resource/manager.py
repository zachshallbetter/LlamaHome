"""Resource management functionality."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Union

import torch

from ..utils import MemoryTracker
from .config import GPUConfig
from .monitor import MonitorConfig, PerformanceMonitor
from .multi_gpu import MultiGPUManager


class ResourceManager:
    """Manages system resources and optimization."""

    def __init__(self, config: GPUConfig) -> None:
        """Initialize resource manager."""
        self.config = config
        self.memory_tracker = MemoryTracker()
        self.monitor = PerformanceMonitor(config=MonitorConfig())
        self.gpu_manager = MultiGPUManager(config)
        self._setup_gpu()
        self._setup_devices()
        self._setup_monitoring()

    def _setup_gpu(self) -> None:
        """Configure GPU settings."""
        if torch.cuda.is_available():
            # Set memory growth
            if self.config.allow_growth:
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)

            # Enable TF32 if configured
            if self.config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def _setup_devices(self) -> None:
        """Set up GPU devices."""
        if torch.cuda.is_available() and self.config.allowed_devices:
            torch.cuda.set_device(self.config.allowed_devices[0])

    def _setup_monitoring(self) -> None:
        """Set up monitoring infrastructure."""
        self.metrics: Dict[str, float] = {}
        self.alerts: Dict[str, bool] = {}

    @asynccontextmanager
    async def optimize(self) -> AsyncGenerator[None, None]:
        """Context manager for resource optimization.

        Yields:
            None
        """
        try:
            # Check if optimization is needed
            if await self.monitor.should_optimize():
                # Perform memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            yield

        finally:
            # Cleanup after operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@dataclass
class GPUConfig:
    """GPU resource configuration."""

    device: Optional[str] = None
    memory_fraction: float = 0.9
    allow_growth: bool = True
    enable_tf32: bool = False
    allowed_devices: Optional[List[int]] = None
    device_map: Optional[Dict[str, Union[int, str]]] = None
    max_memory: Optional[Dict[int, str]] = None
    offload_folder: Optional[str] = None
