"""Configuration schemas."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ModelSchema(BaseModel):
    """Model configuration schema."""

    name: str
    path: Path
    cache_dir: Path
    model_size: str
    model_variant: str = "chat"
    model_quant: str = "f16"
    num_gpu_layers: int = Field(ge=1)
    max_seq_len: int = Field(ge=1)
    max_batch_size: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    context_window: int = Field(ge=1)
    min_gpu_memory: int = Field(ge=1)

    @validator("model_quant")
    def validate_quant(cls, v):
        if v not in {"f16", "f32", "int8", "int4"}:
            raise ValueError(f"Invalid quantization: {v}")
        return v


class TrainingSchema(BaseModel):
    """Training configuration schema."""

    cache: Dict[str, Union[int, bool, str, Path]]
    optimization: Dict[str, Union[float, int, str, bool]]
    resources: Dict[str, Union[float, int]]

    @validator("optimization")
    def validate_optimization(cls, v):
        if not (0 < v["learning_rate"] <= 1.0):
            raise ValueError("Learning rate must be between 0 and 1")
        return v


class DistributedSchema(BaseModel):
    """Distributed configuration schema."""

    basic: Dict[str, Union[str, int]]
    resources: Dict[str, Union[float, int, bool]]
    communication: Dict[str, Union[int, bool]]

    @validator("basic")
    def validate_basic(cls, v):
        if v["rank"] >= v["world_size"]:
            raise ValueError("Rank must be less than world_size")
        return v


class BaseSchema(BaseModel):
    """Base schema with common validation."""

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    @validator("*", pre=True)
    def validate_none_values(cls, v: Any) -> Any:
        """Validate none values."""
        if v is None:
            raise ValueError("None values not allowed")
        return v


class InferenceSchema(BaseSchema):
    """Schema for inference configuration."""

    def validate_model_path(cls, value: str) -> str:
        """Validate model path exists."""
        from pathlib import Path

        path = Path(value)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {value}")
        return value

    def validate_device(cls, value: str) -> str:
        """Validate device is available."""
        import torch

        if value.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        return value

    def validate_batch_size(cls, value: int) -> int:
        """Validate batch size is positive."""
        if value < 1:
            raise ValueError("Batch size must be positive")
        return value

    def validate_max_length(cls, value: int) -> int:
        """Validate max length is positive."""
        if value < 1:
            raise ValueError("Max length must be positive")
        return value

    def validate_temperature(cls, value: float) -> float:
        """Validate temperature is positive."""
        if value <= 0:
            raise ValueError("Temperature must be positive")
        return value

    def validate_top_p(cls, value: float) -> float:
        """Validate top_p is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("Top_p must be between 0 and 1")
        return value

    model_name: str
    model_path: Optional[Path]
    gpu_config: Dict[str, Union[str, int, float, bool]]
    resource: Dict[str, Union[float, int]]
    device: str
    dtype: str
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    model_revision: Optional[str] = None
    quantization: Optional[str] = None
    processing: Dict[str, Union[int, float, bool]]

    @validator("dtype")
    def validate_dtype(cls, v):
        valid_dtypes = {"float16", "float32", "bfloat16", "int8", "int4"}
        if v not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {v}. Must be one of {valid_dtypes}")
        return v


class MonitoringSchema(BaseModel):
    """Monitoring configuration schema."""

    monitoring: Dict[str, Union[int, bool]]

    @validator("monitoring")
    def validate_monitoring(cls, v):
        if v["log_interval"] < 1:
            raise ValueError("Log interval must be positive")
        if v["save_interval"] < 1:
            raise ValueError("Save interval must be positive")
        if v["metrics_history_size"] < 1:
            raise ValueError("Metrics history size must be positive")
        return v


class MetricsSchema(BaseModel):
    """Metrics configuration schema."""

    metrics: Dict[str, Union[List[str], int, str, bool]]

    @validator("metrics")
    def validate_metrics(cls, v):
        valid_formats = {"json", "csv", "parquet"}
        valid_storage = {"file", "database", "memory"}

        if v["export_format"] not in valid_formats:
            raise ValueError(f"Invalid export format: {v['export_format']}")
        if v["storage_type"] not in valid_storage:
            raise ValueError(f"Invalid storage type: {v['storage_type']}")
        if v["aggregation_interval"] < 1:
            raise ValueError("Aggregation interval must be positive")
        if v["retention_days"] < 1:
            raise ValueError("Retention days must be positive")
        return v
