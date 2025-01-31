"""Resource management functionality."""

from dataclasses import dataclass
from typing import AsyncContextManager, Dict, List, Optional, Type, Union

import torch
from torch.utils.data import get_worker_info

from .utils import MemoryTracker

# Use proper type hints for exception handling
ExceptionType = Type[BaseException]


@dataclass
class ResourceConfig:
    """Resource configuration."""

    gpu_memory_fraction: float = 0.9
    cpu_usage_threshold: float = 0.8
    io_queue_size: int = 1000
    max_parallel_requests: int = 10
    timeout: float = 30.0


def get_optimal_device() -> str:
    """Get the optimal available device for the current platform."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GPUConfig:
    """GPU resource configuration."""

    device: Optional[str] = None  # Will be set to optimal device if None
    memory_fraction: float = 0.9
    allow_growth: bool = True
    allowed_devices: Optional[List[int]] = None
    device_map: Optional[Dict[str, Union[int, str]]] = None
    max_memory: Optional[Dict[int, str]] = None
    offload_folder: Optional[str] = None

    def __post_init__(self) -> None:
        """Set optimal device if none specified."""
        if self.device is None:
            self.device = get_optimal_device()


class ResourceOptimizer(AsyncContextManager):
    """Resource optimization context manager."""

    async def __aenter__(self) -> "ResourceOptimizer":
        """Enter context."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[ExceptionType],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Exit context."""
        pass


class ResourceManager:
    """Manages system resources for model operations."""

    def __init__(self, config: GPUConfig) -> None:
        """Initialize resource manager."""
        self.config = config
        self.memory_tracker = MemoryTracker()
        self._setup_devices()
        self._setup_monitoring()

    def _setup_devices(self) -> None:
        """Set up GPU devices."""
        if torch.cuda.is_available():
            if self.config.allowed_devices:
                torch.cuda.set_device(self.config.allowed_devices[0])

    async def optimize(self) -> AsyncContextManager:
        """Optimize resource usage."""
        return self

    def _setup_monitoring(self) -> None:
        """Set up monitoring infrastructure."""
        self.metrics: Dict[str, float] = {}
        self.alerts: Dict[str, bool] = {}

    def _worker_init_fn(self, worker_id: int) -> None:
        """Initialize worker with memory limits."""
        worker_info = get_worker_info()
        if worker_info is not None and hasattr(worker_info.dataset, "memory_limit"):
            worker_info.dataset.memory_limit = self.config.max_memory

    async def get_memory_info(self) -> Dict[int, float]:
        """Get GPU memory usage in GB."""
        memory_info = {}
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                memory_info[device] = torch.cuda.memory_allocated(device) / 1024**3
        return memory_info

    async def __aenter__(self) -> "ResourceManager":
        """Enter context."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[ExceptionType],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Exit context."""
        pass
