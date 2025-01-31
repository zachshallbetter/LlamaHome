"""Tests for model configuration."""

import pytest
from pathlib import Path
from typing import Dict, Any

from src.core.models.config import (
    ModelConfig,
    ModelSpecs,
    ResourceSpecs,
    OptimizationSpecs,
    H2OSpecs,
    SecuritySpecs
)

@pytest.fixture
def test_model_config() -> Dict[str, Any]:
    """Test model configuration data."""
    return {
        "model": {
            "name": "llama",
            "family": "llama",
            "size": "13b",
            "variant": "chat",
            "revision": "main"
        },
        "resources": {
            "min_gpu_memory": 16,
            "max_batch_size": 32,
            "max_sequence_length": 32768,
            "device_map": "auto",
            "torch_dtype": "float16"
        },
        "optimization": {
            "attention_implementation": "h2o",
            "use_bettertransformer": True,
            "use_compile": True,
            "compile_mode": "reduce-overhead"
        },
        "h2o": {
            "enabled": True,
            "window_length": 512,
            "heavy_hitter_tokens": 128,
            "compression": True
        },
        "security": {
            "trust_remote_code": False,
            "use_auth_token": False,
            "verify_downloads": True,
            "allowed_model_sources": ["huggingface.co"]
        }
    }

async def test_model_config_load(
    config_dir: Path,
    test_model_config: Dict[str, Any]
):
    """Test loading model configuration."""
    # Create test config file
    config_path = config_dir / "model_config.toml"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    import toml
    with open(config_path, "w") as f:
        toml.dump(test_model_config, f)
    
    # Load config
    config = await ModelConfig.load(config_dir)
    
    # Verify model specs
    assert config.model.name == "llama"
    assert config.model.family == "llama"
    assert config.model.size == "13b"
    assert config.model.variant == "chat"
    
    # Verify resource specs
    assert config.resources.min_gpu_memory == 16
    assert config.resources.max_batch_size == 32
    assert config.resources.max_sequence_length == 32768
    
    # Verify optimization specs
    assert config.optimization.attention_implementation == "h2o"
    assert config.optimization.use_bettertransformer is True
    assert config.optimization.compile_mode == "reduce-overhead"
    
    # Verify H2O specs
    assert config.h2o.enabled is True
    assert config.h2o.window_length == 512
    assert config.h2o.heavy_hitter_tokens == 128
    
    # Verify security specs
    assert config.security.trust_remote_code is False
    assert config.security.verify_downloads is True
    assert "huggingface.co" in config.security.allowed_model_sources

async def test_model_config_validation():
    """Test model configuration validation."""
    # Test invalid model family
    with pytest.raises(ValueError):
        ModelSpecs(
            name="test",
            family="invalid",
            size="13b"
        )
    
    # Test invalid model size
    with pytest.raises(ValueError):
        ModelSpecs(
            name="test",
            family="llama",
            size="invalid"
        )
    
    # Test invalid GPU memory
    with pytest.raises(ValueError):
        ResourceSpecs(min_gpu_memory=4)
    
    # Test invalid sequence length
    with pytest.raises(ValueError):
        ResourceSpecs(max_sequence_length=256)
    
    # Test invalid attention implementation
    with pytest.raises(ValueError):
        OptimizationSpecs(attention_implementation="invalid")
    
    # Test invalid window length
    with pytest.raises(ValueError):
        H2OSpecs(window_length=64)

async def test_model_paths():
    """Test model path generation."""
    config = ModelConfig(
        model=ModelSpecs(
            name="llama",
            family="llama",
            size="13b"
        ),
        resources=ResourceSpecs(),
        optimization=OptimizationSpecs(),
        h2o=H2OSpecs(),
        security=SecuritySpecs()
    )
    
    # Test model path
    base_path = Path("/models")
    model_path = config.get_model_path(base_path)
    assert model_path == Path("/models/llama/llama-13b")
    
    # Test cache path
    cache_dir = Path("/cache")
    cache_path = config.get_cache_path(cache_dir)
    assert cache_path == Path("/cache/llama/llama-13b") 