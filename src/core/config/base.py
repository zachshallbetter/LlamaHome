from pathlib import Path
from typing import TypeVar

import toml
from pydantic import BaseModel, Field, validator

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base configuration class with common functionality."""

    class Config:
        extra = "forbid"  # Prevent extra attributes
        validate_assignment = True  # Validate on attribute assignment
        arbitrary_types_allowed = True

    @classmethod
    def load_from_file(cls: type[T], path: Path) -> T:
        """Load configuration from a TOML file."""
        try:
            data = toml.load(str(path))
            return cls.parse_obj(data)
        except Exception as e:
            raise ConfigError(f"Failed to load config from {path}: {str(e)}")

    @classmethod
    def load_from_env(cls: type[T], prefix: str = "") -> T:
        """Load configuration from environment variables."""
        import os

        from dotenv import load_dotenv

        load_dotenv()

        env_config = {}
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            clean_key = key.replace(prefix, "").lower()
            env_config[clean_key] = value

        try:
            return cls.parse_obj(env_config)
        except Exception as e:
            raise ConfigError(f"Failed to load config from env: {str(e)}")

    def save_to_file(self, path: Path) -> None:
        """Save configuration to a TOML file."""
        try:
            with open(path, "w") as f:
                toml.dump(self.dict(), f)
        except Exception as e:
            raise ConfigError(f"Failed to save config to {path}: {str(e)}")

    def merge(self, other: "BaseConfig") -> None:
        """Merge another configuration into this one."""
        for key, value in other.dict().items():
            if hasattr(self, key):
                setattr(self, key, value)


class ConfigError(Exception):
    """Configuration related errors."""

    pass


class ResourceConfig(BaseConfig):
    """Resource management configuration."""

    gpu_memory_fraction: float = Field(0.9, ge=0.0, le=1.0)
    cpu_usage_threshold: float = Field(0.8, ge=0.0, le=1.0)
    max_workers: int = Field(4, ge=1)
    io_queue_size: int = Field(1000, ge=1)
    enable_gpu: bool = True
    cuda_devices: list[int] | None = None

    @validator("cuda_devices")
    def validate_cuda_devices(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            import torch

            available = list(range(torch.cuda.device_count()))
            for device in v:
                if device not in available:
                    raise ValueError(f"CUDA device {device} not available")
        return v


class ProcessingConfig(BaseConfig):
    """Data processing configuration."""

    batch_size: int = Field(32, ge=1)
    max_sequence_length: int = Field(512, ge=1)
    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, ge=1)
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True


class OptimizationConfig(BaseConfig):
    """Training optimization configuration."""

    learning_rate: float = Field(5e-5, gt=0.0)
    weight_decay: float = Field(0.01, ge=0.0)
    warmup_steps: int = Field(100, ge=0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    gradient_accumulation_steps: int = Field(1, ge=1)
    mixed_precision: bool = True
    gradient_checkpointing: bool = False


class MonitoringConfig(BaseConfig):
    """System monitoring configuration."""

    enable_monitoring: bool = True
    metrics_interval: int = Field(60, ge=1)
    log_level: str = "INFO"
    enable_profiling: bool = False
    alert_threshold: float = Field(0.9, ge=0.0, le=1.0)
    metrics_history_size: int = Field(1000, ge=1)


class CacheConfig(BaseConfig):
    """Cache management configuration."""

    memory_size: int = Field(1000, ge=0)
    disk_size: int = Field(10000, ge=0)
    cleanup_interval: int = Field(3600, ge=1)
    max_age_days: int = Field(7, ge=1)
    use_mmap: bool = True
    compression: bool = True
    async_writes: bool = True
