"""Resource configuration classes."""

from dataclasses import dataclass


@dataclass
class GPUConfig:
    """GPU resource configuration."""

    memory_fraction: float = 0.9
    allow_growth: bool = True
    per_process_memory: str = "12GB"
    enable_tf32: bool = True


@dataclass
class MonitorConfig:
    """Resource monitoring configuration."""

    check_interval: float = 1.0
    memory_threshold: float = 0.9
    cpu_threshold: float = 0.8
    gpu_temp_threshold: float = 80.0


@dataclass
class ResourceConfig:
    """Resource management configuration."""

    gpu: GPUConfig = GPUConfig()
    monitor: MonitorConfig = MonitorConfig()
    max_parallel_requests: int = 10
    io_queue_size: int = 1000
    timeout: float = 30.0


@dataclass
class MemoryConfig:
    """Memory management configuration."""

    cache_size: str = "4GB"
    min_free: str = "2GB"
    cleanup_margin: float = 0.1
    check_interval: float = 1.0
