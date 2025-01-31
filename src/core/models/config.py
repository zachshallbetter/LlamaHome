"""Model manager configuration."""

from dataclasses import dataclass
from typing import List, Literal, Optional
from pathlib import Path
from pydantic import Field

from ..config import ConfigManager
from ..schemas import ModelSchema
from ..config.base import BaseConfig

ModelFamily = Literal["llama", "gpt4", "claude"]
ModelSize = Literal["7b", "13b", "70b"]
ModelVariant = Literal["base", "chat", "code"]
DeviceMap = Literal["auto", "balanced", "sequential"]
TorchDtype = Literal["float32", "float16", "bfloat16"]
AttentionImpl = Literal["default", "flash", "h2o"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune"]

class ModelSpecs(BaseConfig):
    """Model specifications."""
    name: str
    family: ModelFamily
    size: ModelSize
    variant: ModelVariant = "base"
    revision: str = "main"
    quantization: Optional[str] = None

class ResourceSpecs(BaseConfig):
    """Model resource specifications."""
    min_gpu_memory: int = Field(8, ge=8)
    max_batch_size: int = Field(32, ge=1)
    max_sequence_length: int = Field(32768, ge=512)
    device_map: DeviceMap = "auto"
    torch_dtype: TorchDtype = "float16"

class OptimizationSpecs(BaseConfig):
    """Model optimization specifications."""
    attention_implementation: AttentionImpl = "default"
    use_bettertransformer: bool = False
    use_compile: bool = False
    compile_mode: CompileMode = "default"

class H2OSpecs(BaseConfig):
    """H2O optimization specifications."""
    enabled: bool = False
    window_length: int = Field(512, ge=128)
    heavy_hitter_tokens: int = Field(128, ge=32)
    compression: bool = True

class SecuritySpecs(BaseConfig):
    """Model security specifications."""
    trust_remote_code: bool = False
    use_auth_token: bool = False
    verify_downloads: bool = True
    allowed_model_sources: List[str] = ["huggingface.co"]

class ModelConfig(BaseConfig):
    """Complete model configuration."""
    model: ModelSpecs
    resources: ResourceSpecs
    optimization: OptimizationSpecs
    h2o: H2OSpecs
    security: SecuritySpecs

    @classmethod
    async def load(
        cls,
        config_dir: Path = Path("config"),
        env_prefix: str = "LLAMAHOME_"
    ) -> 'ModelConfig':
        """Load model configuration."""
        from ..config.manager import ConfigManager
        
        manager = ConfigManager(config_dir, env_prefix)
        
        # Load model config
        config = await manager.load_config(
            cls,
            "model",
            "model_config.toml"
        )
        
        return config

    def get_model_path(self, base_path: Path) -> Path:
        """Get the model path."""
        return (
            base_path / 
            self.model.family /
            f"{self.model.name}-{self.model.size}"
        )

    def get_cache_path(self, cache_dir: Path) -> Path:
        """Get the model cache path."""
        return (
            cache_dir / 
            self.model.family /
            f"{self.model.name}-{self.model.size}"
        )
