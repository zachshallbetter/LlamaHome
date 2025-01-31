"""Resource management functionality."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch

from .config import GPUConfig
from .monitor import PerformanceMonitor
from .multi_gpu import MultiGPUManager


class ResourceManager:
    """Manages system resources and optimization."""

    def __init__(self, gpu_config: GPUConfig) -> None:
        """Initialize resource manager.

        Args:
            gpu_config: GPU configuration
        """
        self.gpu_config = gpu_config
        self.monitor = PerformanceMonitor(gpu_config.monitor)
        self.gpu_manager = MultiGPUManager(gpu_config)
        self._setup_gpu()

    def _setup_gpu(self) -> None:
        """Configure GPU settings."""
        if torch.cuda.is_available():
            # Set memory growth
            if self.gpu_config.allow_growth:
                torch.cuda.set_per_process_memory_fraction(
                    self.gpu_config.memory_fraction
                )

            # Enable TF32 if configured
            if self.gpu_config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

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
