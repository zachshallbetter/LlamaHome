from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from ..core.config.base import (
    BaseConfig,
    MonitoringConfig,
    OptimizationConfig,
    ResourceConfig,
)


class CheckpointConfig(BaseConfig):
    """Checkpointing configuration."""

    save_steps: int = Field(1000, ge=1)
    save_total_limit: int = Field(5, ge=1)
    save_strategy: Literal["steps", "epoch"] = "steps"
    evaluation_strategy: Literal["steps", "epoch"] = "steps"
    eval_steps: int = Field(100, ge=1)
    logging_steps: int = Field(10, ge=1)


class DistributedConfig(BaseConfig):
    """Distributed training configuration."""

    backend: Literal["nccl", "gloo"] = "nccl"
    world_size: int = Field(1, ge=1)
    num_nodes: int = Field(1, ge=1)
    node_rank: int = Field(0, ge=0)
    local_rank: int = Field(0, ge=0)
    master_addr: str = "localhost"
    master_port: str = "29500"
    sync_batch_norm: bool = True


class DataConfig(BaseConfig):
    """Training data configuration."""

    batch_size: int = Field(32, ge=1)
    max_sequence_length: int = Field(512, ge=1)
    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, ge=1)
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True
    validation_split: float = Field(0.1, ge=0.0, le=1.0)


class TrainingConfig(BaseConfig):
    """Complete training configuration."""

    optimization: OptimizationConfig
    data: DataConfig
    resources: ResourceConfig
    monitoring: MonitoringConfig
    checkpointing: CheckpointConfig
    distributed: Optional[DistributedConfig] = None

    @classmethod
    async def load(
        cls, config_dir: Path = Path("config"), env_prefix: str = "LLAMAHOME_"
    ) -> "TrainingConfig":
        """Load training configuration."""
        from ..core.config.manager import ConfigManager

        manager = ConfigManager(config_dir, env_prefix)

        # Load optimization config
        optimization = await manager.load_config(
            OptimizationConfig, "optimization", "optimization_config.toml"
        )

        # Load data config
        data = await manager.load_config(DataConfig, "data", "data_config.toml")

        # Load resource config
        resources = await manager.load_config(
            ResourceConfig, "resources", "resource_config.toml"
        )

        # Load monitoring config
        monitoring = await manager.load_config(
            MonitoringConfig, "monitoring", "monitoring_config.toml"
        )

        # Load checkpointing config
        checkpointing = await manager.load_config(
            CheckpointConfig, "checkpointing", "checkpointing_config.toml"
        )

        # Load distributed config if enabled
        distributed = None
        if resources.enable_gpu and resources.cuda_devices:
            distributed = await manager.load_config(
                DistributedConfig, "distributed", "distributed_config.toml"
            )

        return cls(
            optimization=optimization,
            data=data,
            resources=resources,
            monitoring=monitoring,
            checkpointing=checkpointing,
            distributed=distributed,
        )
