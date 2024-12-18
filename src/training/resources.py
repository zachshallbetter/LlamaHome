"""
Resource management implementation for training pipeline.
"""

import asyncio
import psutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch



@dataclass


class ResourceConfig:
    """Resource configuration."""
    gpu_memory_fraction: float = 0.9
    cpu_usage_threshold: float = 0.8
    io_queue_size: int = 1000
    wait_interval: float = 0.1


class Resource(ABC):
    """Abstract base class for resources."""


    def __init__(self):
        self._lock = asyncio.Lock()
        self._in_use = False

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if resource is available."""
        pass

    async def wait(self) -> None:
        """Wait for resource availability."""
        while not await self.is_available():
            await asyncio.sleep(ResourceConfig.wait_interval)
        async with self._lock:
            self._in_use = True


    def release(self) -> None:
        """Release resource."""
        self._in_use = False


class GPUResource(Resource):
    """GPU memory and compute resource."""


    def __init__(self, memory_fraction: float = ResourceConfig.gpu_memory_fraction):
        super().__init__()
        self.memory_fraction = memory_fraction
        self._setup_gpu()


    def _setup_gpu(self) -> None:
        """Initialize GPU monitoring."""
        if not torch.cuda.is_available():
            raise ResourceError("No GPU available")
        self.device = torch.device("cuda")
        self.total_memory = torch.cuda.get_device_properties(0).total_memory

    async def is_available(self) -> bool:
        """Check GPU memory availability."""
        if self._in_use:
            return False

        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()

        available_fraction = 1.0 - (
            (memory_allocated + memory_reserved) / self.total_memory
        )
        return available_fraction >= (1.0 - self.memory_fraction)


    def get_memory_info(self) -> dict:
        """Get detailed GPU memory information."""
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "total": self.total_memory,
            "available": self.total_memory - torch.cuda.memory_allocated()
        }


class CPUResource(Resource):
    """CPU utilization resource."""


    def __init__(self, usage_threshold: float = ResourceConfig.cpu_usage_threshold):
        super().__init__()
        self.usage_threshold = usage_threshold
        self._setup_cpu()


    def _setup_cpu(self) -> None:
        """Initialize CPU monitoring."""
        self.cpu_count = psutil.cpu_count()
        self._usage_history = []

    async def is_available(self) -> bool:
        """Check CPU availability."""
        if self._in_use:
            return False

        cpu_percent = psutil.cpu_percent()
        self._update_history(cpu_percent)

        return self._get_average_usage() <= (self.usage_threshold * 100)


    def _update_history(self, usage: float) -> None:
        """Update CPU usage history."""
        self._usage_history.append(usage)
        if len(self._usage_history) > 10:
            self._usage_history.pop(0)


    def _get_average_usage(self) -> float:
        """Get average CPU usage."""
        if not self._usage_history:
            return 0.0
        return sum(self._usage_history) / len(self._usage_history)


    def get_cpu_info(self) -> dict:
        """Get detailed CPU information."""
        return {
            "count": self.cpu_count,
            "usage": psutil.cpu_percent(percpu=True),
            "average": self._get_average_usage(),
            "memory": dict(psutil.virtual_memory()._asdict())
        }


class IOResource(Resource):
    """I/O bandwidth resource."""


    def __init__(self, queue_size: int = ResourceConfig.io_queue_size):
        super().__init__()
        self.queue_size = queue_size
        self._setup_io()


    def _setup_io(self) -> None:
        """Initialize I/O monitoring."""
        self._io_counters = psutil.disk_io_counters()
        self._queue = asyncio.Queue(maxsize=self.queue_size)

    async def is_available(self) -> bool:
        """Check I/O availability."""
        if self._in_use:
            return False

        return self._queue.qsize() < self.queue_size

    async def queue_operation(self, operation: callable) -> None:
        """Queue an I/O operation."""
        await self._queue.put(operation)
        try:
            result = await operation()
            return result
        finally:
            self._queue.get_nowait()


    def get_io_info(self) -> dict:
        """Get detailed I/O information."""
        current = psutil.disk_io_counters()
        return {
            "read_bytes": current.read_bytes - self._io_counters.read_bytes,
            "write_bytes": current.write_bytes - self._io_counters.write_bytes,
            "queue_size": self._queue.qsize(),
            "queue_capacity": self.queue_size
        }


class ResourceMonitor:
    """Resource monitoring and management."""


    def __init__(self, config: Optional[ResourceConfig] = None):
        self.config = config or ResourceConfig()
        self.resources = {
            "gpu": GPUResource(self.config.gpu_memory_fraction),
            "cpu": CPUResource(self.config.cpu_usage_threshold),
            "io": IOResource(self.config.io_queue_size)
        }

    async def wait_for_resources(self) -> None:
        """Wait for all resources to be available."""
        await asyncio.gather(*[
            resource.wait()
            for resource in self.resources.values()
        ])


    def release_resources(self) -> None:
        """Release all resources."""
        for resource in self.resources.values():
            resource.release()


    def get_resource_info(self) -> dict:
        """Get detailed information about all resources."""
        return {
            "gpu": self.resources["gpu"].get_memory_info(),
            "cpu": self.resources["cpu"].get_cpu_info(),
            "io": self.resources["io"].get_io_info()
        }


class ResourceError(Exception):
    """Resource management error."""
    pass
