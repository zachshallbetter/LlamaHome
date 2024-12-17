"""Integration tests for core system components."""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.model import HybridModel
from src.core.attention import HybridAttention
from src.core.config_handler import ConfigManager
from utils.model_manager import ModelManager


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment with necessary directories and files."""
    # Create config directory
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
def mock_config():
    """Create mock configuration."""
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


class TestCoreIntegration:
    """Integration tests for core components."""
    
    @pytest.mark.integration
    def test_model_initialization_flow(self, setup_test_env, mock_config):
        """Test complete model initialization flow."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        model_manager = ModelManager(config_manager)
        
        with patch('src.core.model.HybridModel') as mock_hybrid_model:
            # Setup mock model instance
            mock_instance = MagicMock()
            mock_hybrid_model.return_value = mock_instance
            
            # Test model loading
            model = model_manager.load_model("llama", "3.3-7b")
            
            # Verify initialization chain
            mock_hybrid_model.assert_called_once()
            assert isinstance(model, MagicMock)  # In this case, our mock instance
    
    @pytest.mark.integration
    def test_attention_integration(self, setup_test_env, mock_config):
        """Test attention mechanism integration."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        with patch('transformers.AutoModelForCausalLM') as mock_base_model:
            with patch('transformers.AutoTokenizer') as mock_tokenizer:
                # Setup mock instances
                mock_base_model.from_pretrained.return_value = MagicMock()
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                
                # Initialize model with attention
                model = HybridModel(
                    model=mock_base_model.from_pretrained("llama"),
                    tokenizer=mock_tokenizer.from_pretrained("llama"),
                    config=mock_config
                )
                
                # Verify attention initialization
                assert isinstance(model.attention, HybridAttention)
                
                # Test attention integration
                input_ids = torch.randint(0, 32000, (1, 128))
                attention_mask = torch.ones_like(input_ids)
                
                with patch.object(model.attention, 'forward') as mock_attention_forward:
                    model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=256
                    )
                    mock_attention_forward.assert_called()
    
    @pytest.mark.integration
    def test_config_model_integration(self, setup_test_env, mock_config):
        """Test configuration and model integration."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        # Update configuration
        config_manager.update_training_config({
            "batch_size": 16,
            "learning_rate": 2e-5
        })
        
        with patch('src.core.model.HybridModel') as mock_hybrid_model:
            # Setup mock model instance
            mock_instance = MagicMock()
            mock_hybrid_model.return_value = mock_instance
            
            model_manager = ModelManager(config_manager)
            model = model_manager.load_model("llama", "3.3-7b")
            
            # Verify configuration was properly passed
            mock_hybrid_model.assert_called_once()
            call_args = mock_hybrid_model.call_args[1]
            assert "config" in call_args
    
    @pytest.mark.integration
    def test_end_to_end_generation(self, setup_test_env, mock_config):
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
    def test_model_persistence(self, setup_test_env, mock_config):
        """Test model state persistence and loading."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        model_manager = ModelManager(config_manager)
        
        with patch('src.core.model.HybridModel') as mock_hybrid_model:
            # Setup mock instance
            mock_instance = MagicMock()
            mock_hybrid_model.return_value = mock_instance
            
            # Test save and load operations
            model = model_manager.load_model("llama", "3.3-7b")
            save_path = setup_test_env / "models" / "test_save.pt"
            
            model.save(save_path)
            mock_instance.save.assert_called_once_with(save_path)
            
            model_manager.load_model("llama", "3.3-7b", model_path=save_path)
            mock_hybrid_model.assert_called()
    
    @pytest.mark.integration
    def test_performance_monitoring(self, setup_test_env, mock_config):
        """Test performance monitoring integration."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        with patch('transformers.AutoModelForCausalLM') as mock_base_model:
            with patch('transformers.AutoTokenizer') as mock_tokenizer:
                # Setup mock instances
                mock_base_model.from_pretrained.return_value = MagicMock()
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                
                model = HybridModel(
                    model=mock_base_model.from_pretrained("llama"),
                    tokenizer=mock_tokenizer.from_pretrained("llama"),
                    config=mock_config
                )
                
                with patch('src.training.monitoring.MetricsCollector') as mock_collector:
                    # Test generation with monitoring
                    input_ids = torch.randint(0, 32000, (1, 128))
                    model.generate(
                        input_ids=input_ids,
                        max_length=256,
                        monitor_performance=True
                    )
                    mock_collector.assert_called()
    
    @pytest.mark.integration
    def test_error_handling(self, setup_test_env, mock_config):
        """Test error handling across components."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        model_manager = ModelManager(config_manager)
        
        # Test invalid model loading
        with pytest.raises(ValueError):
            model_manager.load_model("invalid_model", "invalid_version")
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            config_manager.update_training_config({"batch_size": -1})
        
        # Test model initialization with invalid config
        invalid_config = MagicMock()
        invalid_config.models = {}
        
        with pytest.raises(ValueError):
            model_manager = ModelManager(invalid_config)
            model_manager.load_model("llama", "3.3-7b")
    
    @pytest.mark.integration
    def test_resource_management(self, setup_test_env, mock_config):
        """Test resource management across components."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        with patch('transformers.AutoModelForCausalLM') as mock_base_model:
            with patch('transformers.AutoTokenizer') as mock_tokenizer:
                # Setup mock instances
                mock_base_model.from_pretrained.return_value = MagicMock()
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                
                model = HybridModel(
                    model=mock_base_model.from_pretrained("llama"),
                    tokenizer=mock_tokenizer.from_pretrained("llama"),
                    config=mock_config
                )
                
                # Test cleanup
                with patch.object(model.model, 'cpu') as mock_cpu:
                    model.cleanup()
                    mock_cpu.assert_called_once()
                
                # Test cache cleanup
                assert not model.attention.has_cache()
