"""Configuration management."""

import os
import tomli
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Constants
CONFIG_DIR = Path("config")
DATA_DIR = Path(".data")

@dataclass
class PathConfig:
    """Path configuration."""
    
    root: Path = DATA_DIR
    paths: Dict[str, Path] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize standard paths."""
        self.paths.update({
            "models": self.root / "models",
            "cache": self.root / "cache",
            "training": self.root / "training",
            "telemetry": self.root / "telemetry",
            "memory": self.root / "memory",
            "logs": self.root / "logs",
            "local": self.root / "local",
            "temp": self.root / "temp"
        })
        
        # Create directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Path:
        """Get path by key."""
        return self.paths[key]

class ConfigManager:
    """Configuration manager."""
    
    def __init__(self) -> None:
        self.paths = PathConfig()
        self.configs: Dict[str, Dict[str, Any]] = {}
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load all TOML configurations."""
        config_files = {
            "model": CONFIG_DIR / "model_config.toml",
            "training": CONFIG_DIR / "training_config.toml",
            "distributed": CONFIG_DIR / "distributed_config.toml",
            "system": CONFIG_DIR / "system_commands.toml",
            "code_check": CONFIG_DIR / "code_check.toml",
            "types": CONFIG_DIR / "llamahome.types.ini"
        }
        
        for name, path in config_files.items():
            if path.exists():
                if path.suffix == ".toml":
                    self.configs[name] = load_toml_config(path)
                # Add handlers for other config types if needed

@dataclass
class ModelConfig:
    """Model configuration."""
    
    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.paths = self.config_manager.paths
        self.model_config = self.config_manager.configs["model"]
        
        # Environment variables take precedence
        self.name = os.getenv("DEFAULT_MODEL_NAME", "llama3.3-7b")
        self.path = Path(os.getenv("LLAMAHOME_MODEL_PATH", str(self.paths.get("models"))))
        self.cache_dir = Path(os.getenv("MODEL_CACHE_DIR", str(self.paths.get("cache") / "models")))
        
        # Load model-specific config from TOML
        model_base = self.name.split("-")[0]
        if model_base in self.model_config["models"]:
            self._load_model_config(model_base)
    
    def _load_model_config(self, model_base: str) -> None:
        """Load model-specific configuration."""
        model_cfg = self.model_config["models"][model_base]
        
        # Basic settings
        self.max_tokens = model_cfg.get("max_tokens", 32768)
        self.context_window = model_cfg.get("context_window", 8192)
        
        # H2O settings
        h2o_cfg = model_cfg.get("h2o_config", {})
        self.h2o_enabled = str_to_bool(os.getenv("LLAMA_H2O_ENABLED", str(h2o_cfg.get("enabled", False))))
        self.h2o_window_length = int(os.getenv("LLAMA_H2O_WINDOW_LENGTH", h2o_cfg.get("window_length", 512)))
        
        # Model size specific settings
        size = self.name.split("-")[1]
        self.min_gpu_memory = model_cfg.get("min_gpu_memory", {}).get(size, 8)
        
        # Llama specific settings
        self.model_size = os.getenv("LLAMA_MODEL_SIZE", size)
        self.model_variant = os.getenv("LLAMA_MODEL_VARIANT", "chat")
        self.model_quant = os.getenv("LLAMA_MODEL_QUANT", "f16")
        self.num_gpu_layers = int(os.getenv("LLAMA_NUM_GPU_LAYERS", "32"))
        self.max_seq_len = int(os.getenv("LLAMA_MAX_SEQ_LEN", "32768"))
        self.max_batch_size = int(os.getenv("LLAMA_MAX_BATCH_SIZE", "8"))

@dataclass
class TrainingConfig:
    """Training configuration."""
    
    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.paths = self.config_manager.paths
        self.training_config = self.config_manager.configs["training"]
        
        # Cache settings
        cache_cfg = self.training_config["cache"]
        self.cache = {
            "memory_size": int(os.getenv("LLAMAHOME_CACHE_SIZE", cache_cfg["memory_size"])),
            "disk_size": cache_cfg["disk_size"],
            "cleanup_interval": cache_cfg["cleanup_interval"],
            "use_mmap": cache_cfg["use_mmap"],
            "compression": cache_cfg["compression"],
            "cache_dir": self.paths.get("cache") / "training"
        }
        
        # Optimization settings
        opt_cfg = self.training_config["optimization"]
        self.optimization = {
            "learning_rate": float(opt_cfg["learning_rate"]),
            "weight_decay": float(opt_cfg["weight_decay"]),
            "warmup_steps": int(opt_cfg["warmup_steps"]),
            "scheduler_type": opt_cfg["scheduler_type"],
            "max_grad_norm": float(opt_cfg["max_grad_norm"]),
            "mixed_precision": opt_cfg["mixed_precision"],
            "gradient_checkpointing": opt_cfg["gradient_checkpointing"]
        }
        
        # Resource settings
        res_cfg = self.training_config["resource"]
        self.resources = {
            "gpu_memory_fraction": float(res_cfg["gpu_memory_fraction"]),
            "cpu_usage_threshold": float(res_cfg["cpu_usage_threshold"]),
            "io_queue_size": int(res_cfg["io_queue_size"])
        }
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        self._validate_ranges()
        self._validate_paths()
        self._validate_dependencies()
    
    def _validate_ranges(self) -> None:
        """Validate numeric ranges."""
        validations = [
            (0.0 < self.optimization["learning_rate"] <= 1.0, "Learning rate must be between 0 and 1"),
            (0.0 <= self.optimization["weight_decay"] <= 1.0, "Weight decay must be between 0 and 1"),
            (self.optimization["warmup_steps"] >= 0, "Warmup steps must be non-negative"),
            (0.0 < self.resources["gpu_memory_fraction"] <= 1.0, "GPU memory fraction must be between 0 and 1")
        ]
        
        for condition, message in validations:
            if not condition:
                raise ConfigError(message)
    
    def _validate_paths(self) -> None:
        """Validate path configurations."""
        cache_dir = self.cache["cache_dir"]
        if not cache_dir.parent.exists():
            raise ConfigError(f"Parent directory for cache does not exist: {cache_dir.parent}")
    
    def _validate_dependencies(self) -> None:
        """Validate configuration dependencies."""
        if self.optimization["mixed_precision"] and not torch.cuda.is_available():
            raise ConfigError("Mixed precision training requires CUDA")

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    
    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.distributed_config = self.config_manager.configs["distributed"]
        
        # Basic distributed settings
        dist_cfg = self.distributed_config["distributed"]
        self.basic = {
            "backend": dist_cfg["backend"],
            "init_method": dist_cfg["init_method"],
            "world_size": int(os.getenv("WORLD_SIZE", dist_cfg["world_size"])),
            "rank": int(os.getenv("RANK", dist_cfg["rank"])),
            "local_rank": int(os.getenv("LOCAL_RANK", dist_cfg["local_rank"]))
        }
        
        # Resource settings
        res_cfg = self.distributed_config["resources"]
        self.resources = {
            "gpu_memory_fraction": float(res_cfg["gpu_memory_fraction"]),
            "gpu_batch_size": int(res_cfg["gpu_batch_size"]),
            "gpu_workers": int(res_cfg["gpu_workers"]),
            "pin_memory": res_cfg["pin_memory"]
        }
        
        # Communication settings
        comm_cfg = self.distributed_config["communication"]
        self.communication = {
            "timeout": int(comm_cfg["timeout"]),
            "broadcast_buffers": comm_cfg["broadcast_buffers"],
            "bucket_cap_mb": int(comm_cfg["bucket_cap_mb"])
        }
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate distributed configuration."""
        self._validate_backend()
        self._validate_ranks()
        self._validate_resources()
    
    def _validate_backend(self) -> None:
        """Validate backend configuration."""
        valid_backends = {"nccl", "gloo", "mpi"}
        if self.basic["backend"] not in valid_backends:
            raise ConfigError(f"Invalid backend: {self.basic['backend']}. Must be one of {valid_backends}")
        
        if self.basic["backend"] == "nccl" and not torch.cuda.is_available():
            raise ConfigError("NCCL backend requires CUDA")
    
    def _validate_ranks(self) -> None:
        """Validate rank configurations."""
        if self.basic["rank"] >= self.basic["world_size"]:
            raise ConfigError(f"Rank {self.basic['rank']} must be less than world_size {self.basic['world_size']}")
        
        if self.basic["local_rank"] >= torch.cuda.device_count():
            raise ConfigError(f"Local rank {self.basic['local_rank']} exceeds available GPU count")
    
    def _validate_resources(self) -> None:
        """Validate resource configurations."""
        if self.resources["gpu_memory_fraction"] > 1.0:
            raise ConfigError("GPU memory fraction cannot exceed 1.0")

def str_to_bool(value: str) -> bool:
    """Convert string to boolean."""
    return value.lower() in ("true", "1", "yes", "on")

class ConfigError(Exception):
    """Configuration error."""
    pass

def load_toml_config(config_path: Path) -> Dict[str, Any]:
    """Load TOML configuration file.
    
    Args:
        config_path: Path to TOML config file
        
    Returns:
        Dict containing configuration
        
    Raises:
        ConfigError: If config file cannot be loaded
    """
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        raise ConfigError(f"Failed to load config {config_path}: {e}") 