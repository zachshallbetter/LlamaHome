"""Resource configuration classes."""

from typing import Optional, List
from pydantic import Field

from ..config.base import BaseConfig

class GPUConfig(BaseConfig):
    """GPU resource configuration."""
    memory_fraction: float = Field(0.9, ge=0.0, le=1.0)
    allow_growth: bool = True
    per_process_memory: str = "12GB"
    enable_tf32: bool = True
    cuda_devices: Optional[List[int]] = None
    
    @property
    def available_memory(self) -> int:
        """Get available GPU memory in bytes."""
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(0).total_memory

class MonitorConfig(BaseConfig):
    """Resource monitoring configuration."""
    check_interval: float = Field(1.0, gt=0.0)
    memory_threshold: float = Field(0.9, ge=0.0, le=1.0)
    cpu_threshold: float = Field(0.8, ge=0.0, le=1.0)
    gpu_temp_threshold: float = Field(80.0, ge=0.0)
    alert_on_threshold: bool = True
    collect_metrics: bool = True

class ResourceConfig(BaseConfig):
    """Complete resource configuration."""
    gpu: GPUConfig
    monitor: MonitorConfig
    max_workers: int = Field(4, ge=1)
    io_queue_size: int = Field(1000, ge=1)
    pin_memory: bool = True
    
    @classmethod
    async def load(cls, config_dir: str = "config") -> 'ResourceConfig':
        """Load resource configuration."""
        from ..config.manager import ConfigManager
        
        manager = ConfigManager(config_dir)
        return await manager.load_config(
            cls,
            "resources",
            "resource_config.toml"
        )
