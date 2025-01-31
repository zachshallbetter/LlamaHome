"""
Configuration classes for inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field

from ..core.config.base import BaseConfig, ProcessingConfig, ResourceConfig
from ..core.schemas import InferenceSchema


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


class ModelConfig(BaseConfig):
    """Model-specific configuration."""

    model_name: str
    model_path: Optional[Path] = None
    trust_remote_code: bool = False
    use_auth_token: bool = False
    model_revision: str = "main"
    quantization: Optional[str] = None
    device_map: Optional[str] = "auto"
    torch_dtype: Optional[str] = "float16"
    max_memory: Optional[dict] = None


class InferenceConfig(BaseConfig):
    """Inference configuration."""

    model: ModelConfig
    resources: ResourceConfig
    processing: ProcessingConfig

    # Inference-specific settings
    max_new_tokens: int = Field(512, ge=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.1, ge=0.0)
    length_penalty: float = Field(1.0, ge=0.0)
    no_repeat_ngram_size: int = Field(3, ge=0)
    num_return_sequences: int = Field(1, ge=1)
    do_sample: bool = True
    early_stopping: bool = True

    # Streaming settings
    stream_output: bool = False
    chunk_size: int = Field(4, ge=1)
    max_chunks: Optional[int] = None

    # Cache settings
    use_cache: bool = True
    cache_dir: Optional[Path] = None

    @classmethod
    async def load(
        cls, config_dir: Path = Path("config"), env_prefix: str = "LLAMAHOME_"
    ) -> "InferenceConfig":
        """Load inference configuration."""
        from ..core.config.manager import ConfigManager

        manager = ConfigManager(config_dir, env_prefix)
        return await manager.load_config(cls, "inference", "inference_config.toml")
