"""
Configuration classes for inference.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union


@dataclass
class CacheConfig:
    """Cache configuration for inference."""

    memory_size: int = 1000  # MB
    disk_size: int = 10000  # MB
    use_mmap: bool = True
    compression: bool = True
    cache_dir: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Processing configuration for inference."""

    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    batch_size: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    early_stopping: bool = True


@dataclass
class ResourceConfig:
    """Resource configuration for inference."""

    gpu_memory_fraction: float = 0.9
    cpu_usage_threshold: float = 0.8
    io_queue_size: int = 1000
    max_parallel_requests: int = 10
    timeout: float = 30.0  # seconds


@dataclass
class InferenceConfig:
    """Main inference configuration."""

    cache: CacheConfig = CacheConfig()
    processing: ProcessingConfig = ProcessingConfig()
    resource: ResourceConfig = ResourceConfig()
    model_revision: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = "float16"
    quantization: Optional[str] = None  # "int8", "int4", etc.
