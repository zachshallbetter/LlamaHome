"""Tests for hybrid model implementation combining H2O and llama-recipes functionality."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import toml
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.core.models import HybridModel
from src.core.attention import HybridAttention
from src.core.config import ConfigManager
from src.core.models import ModelManager


@pytest.fixture(autouse=True)
def setup_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Configure logging for tests."""
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("utils.model_manager")
    logger.setLevel(logging.WARNING)
    yield


@pytest.fixture
def mock_config() -> MagicMock:
    """Fixture providing test model configuration."""
    config = MagicMock()
    config.models = {
        "llama": {
            "versions": {
                "3.3-7b": {
                    "name": "Llama 3.3 7B",
                    "requires_gpu": True,
                    "min_gpu_memory": 12,
                    "h2o_config": {
                        "window_length": 1024,
                        "heavy_hitter_tokens": 256
                    },
                    "type": "base",
                    "format": "meta"
                }
            }
        }
    }
    return config


@pytest.fixture
def setup_test_env(tmp_path: Path, mock_config: MagicMock) -> Path:
    """Set up test environment with temporary directories and config."""
    # Create config directory and file
    config_dir = tmp_path / ".config"
    config_dir.mkdir(parents=True)
    
    # Create models directory
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    
    # Create cache directory
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True)
    
    return tmp_path


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.vocab_size = 32000
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock base model for testing."""
    model = MagicMock(spec=AutoModelForCausalLM)
    model.config.vocab_size = 32000
    model.config.hidden_size = 4096
    return model


class TestHybridModel:
    """Test suite for HybridModel implementation."""
    
    def test_initialization(self, mock_config, mock_tokenizer, mock_model):
        """Test HybridModel initialization with both H2O and llama-recipes components."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        assert hybrid_model.model == mock_model
        assert hybrid_model.tokenizer == mock_tokenizer
        assert isinstance(hybrid_model.attention, HybridAttention)
    
    def test_attention_mechanism(self, mock_config, mock_tokenizer, mock_model):
        """Test hybrid attention mechanism combining H2O and transformer attention."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        # Mock input for attention test
        input_ids = torch.randint(0, 32000, (1, 128))
        attention_mask = torch.ones_like(input_ids)
        
        with patch.object(hybrid_model.attention, 'forward') as mock_attention:
            hybrid_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256
            )
            mock_attention.assert_called()
    
    def test_model_configuration(self, mock_config, mock_tokenizer, mock_model):
        """Test model configuration handling and validation."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        assert hybrid_model.config.models["llama"]["versions"]["3.3-7b"]["h2o_config"]["window_length"] == 1024
        assert hybrid_model.config.models["llama"]["versions"]["3.3-7b"]["h2o_config"]["heavy_hitter_tokens"] == 256
    
    def test_performance_monitoring(self, mock_config, mock_tokenizer, mock_model):
        """Test model performance monitoring capabilities."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        with patch('src.training.monitoring.MetricsCollector') as mock_collector:
            # Test generation with monitoring
            input_ids = torch.randint(0, 32000, (1, 128))
            hybrid_model.generate(
                input_ids=input_ids,
                max_length=256,
                monitor_performance=True
            )
            mock_collector.assert_called()
    
    def test_model_versioning(self, mock_config, mock_tokenizer, mock_model):
        """Test model versioning and compatibility checks."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        assert hybrid_model.get_version() == "3.3-7b"
        assert hybrid_model.is_compatible(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def test_generation_config(self, mock_config, mock_tokenizer, mock_model):
        """Test generation configuration and parameter handling."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        # Test custom generation parameters
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "max_length": 256
        }
        
        input_ids = torch.randint(0, 32000, (1, 128))
        
        with patch.object(mock_model, 'generate') as mock_generate:
            hybrid_model.generate(input_ids=input_ids, **params)
            mock_generate.assert_called_once()
            
            # Verify parameters were passed correctly
            call_args = mock_generate.call_args[1]
            assert call_args["temperature"] == 0.7
            assert call_args["top_p"] == 0.9
            assert call_args["repetition_penalty"] == 1.2
            assert call_args["max_length"] == 256
    
    def test_error_handling(self, mock_config, mock_tokenizer, mock_model):
        """Test error handling and graceful fallbacks."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        # Test invalid input handling
        with pytest.raises(ValueError):
            hybrid_model.generate(input_ids=None)
        
        # Test device compatibility error handling
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError):
                hybrid_model.to("cuda")
    
    def test_resource_management(self, mock_config, mock_tokenizer, mock_model):
        """Test resource management and cleanup."""
        hybrid_model = HybridModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config
        )
        
        # Test proper resource cleanup
        with patch.object(mock_model, 'cpu') as mock_cpu:
            hybrid_model.cleanup()
            mock_cpu.assert_called_once()


class TestModelIntegration:
    """Integration tests for model functionality."""
    
    @pytest.mark.integration
    def test_end_to_end_generation(self, mock_config, setup_test_env):
        """Test end-to-end text generation workflow."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        model_manager = ModelManager(config_manager)
        
        with patch('src.core.model.HybridModel') as mock_hybrid_model:
            # Setup mock response
            mock_instance = MagicMock()
            mock_instance.generate.return_value = "Test response"
            mock_hybrid_model.return_value = mock_instance
            
            # Test generation workflow
            model = model_manager.load_model("llama", "3.3-7b")
            response = model.generate(prompt="Test prompt")
            
            assert response == "Test response"
            mock_instance.generate.assert_called_once()
    
    @pytest.mark.integration
    def test_model_persistence(self, mock_config, setup_test_env):
        """Test model state persistence and loading."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        model_manager = ModelManager(config_manager)
        
        with patch('src.core.model.HybridModel') as mock_hybrid_model:
            # Test save and load operations
            mock_instance = MagicMock()
            mock_hybrid_model.return_value = mock_instance
            
            model = model_manager.load_model("llama", "3.3-7b")
            save_path = setup_test_env / "models" / "test_save.pt"
            
            model.save(save_path)
            mock_instance.save.assert_called_once_with(save_path)
            
            model_manager.load_model("llama", "3.3-7b", model_path=save_path)
            mock_hybrid_model.assert_called()
