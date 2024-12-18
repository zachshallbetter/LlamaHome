"""Memory usage tracking and monitoring."""

import gc
import psutil
from typing import Dict

import torch

from ..utils import LogManager, LogTemplates


class MemoryTracker:
    """Tracks memory usage across CPU and GPU."""


    def __init__(self):
        """Initialize memory tracker."""
        self.logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
        self.process = psutil.Process()


    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics.

        Returns:
            Dictionary of memory statistics
        """
        stats = {
            'ram_used_gb': self._get_ram_usage(),
            'ram_percent': self._get_ram_percent()
        }

        if torch.cuda.is_available():
            stats.update(self._get_gpu_stats())

        return stats


    def _get_ram_usage(self) -> float:
        """Get RAM usage in GB.

        Returns:
            RAM usage in gigabytes
        """
        return self.process.memory_info().rss / (1024 ** 3)


    def _get_ram_percent(self) -> float:
        """Get RAM usage as percentage.

        Returns:
            RAM usage percentage
        """
        return self.process.memory_percent()


    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics.

        Returns:
            Dictionary of GPU memory statistics
        """
        stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            stats.update({
                f'gpu_{i}_allocated_gb': allocated,
                f'gpu_{i}_reserved_gb': reserved
            })
        return stats


    def clear_memory(self) -> None:
        """Clear unused memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
