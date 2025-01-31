"""Tests for configuration management system."""

import os
from pathlib import Path
import pytest
import toml
from unittest.mock import patch, mock_open

from src.core.config import ConfigManager


@pytest.fixture
def mock_config_data():
    """Provide mock configuration data."""
    return {
        "models": {
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
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0
        },
        "system": {
            "cache_dir": ".cache",
            "log_level": "INFO",
            "use_gpu": True
        }
    }


@pytest.fixture
def setup_test_env(tmp_path, mock_config_data):
    """Set up test environment with configuration files."""
    config_dir = tmp_path / ".config"
    config_dir.mkdir(parents=True)
    
    # Create main config file
    config_file = config_dir / "config.toml"
    with open(config_file, "w") as f:
        toml.dump(mock_config_data, f)
    
    return tmp_path


class TestConfigManager:
    """Test suite for configuration management."""
    
    def test_initialization(self, setup_test_env):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        assert config_manager.config is not None
        assert "models" in config_manager.config
        assert "training" in config_manager.config
        assert "system" in config_manager.config
    
    def test_model_config_access(self, setup_test_env):
        """Test access to model configuration."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        model_config = config_manager.get_model_config("llama", "3.3-7b")
        assert model_config["name"] == "Llama 3.3 7B"
        assert model_config["requires_gpu"] is True
        assert model_config["h2o_config"]["window_length"] == 1024
    
    def test_training_config_access(self, setup_test_env):
        """Test access to training configuration."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        training_config = config_manager.get_training_config()
        assert training_config["batch_size"] == 32
        assert training_config["learning_rate"] == 1e-4
        assert training_config["gradient_accumulation_steps"] == 4
    
    def test_system_config_access(self, setup_test_env):
        """Test access to system configuration."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        system_config = config_manager.get_system_config()
        assert system_config["cache_dir"] == ".cache"
        assert system_config["log_level"] == "INFO"
        assert system_config["use_gpu"] is True
    
    def test_config_validation(self, setup_test_env):
        """Test configuration validation."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        # Test invalid model access
        with pytest.raises(ValueError):
            config_manager.get_model_config("invalid_model", "invalid_version")
        
        # Test invalid config section access
        with pytest.raises(KeyError):
            config_manager.config["invalid_section"]
    
    def test_config_update(self, setup_test_env):
        """Test configuration update functionality."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        # Update training configuration
        new_training_config = {
            "batch_size": 64,
            "learning_rate": 2e-4
        }
        config_manager.update_training_config(new_training_config)
        
        # Verify updates
        updated_config = config_manager.get_training_config()
        assert updated_config["batch_size"] == 64
        assert updated_config["learning_rate"] == 2e-4
        
        # Verify other settings remained unchanged
        assert updated_config["gradient_accumulation_steps"] == 4
    
    def test_config_persistence(self, setup_test_env):
        """Test configuration persistence."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        # Update configuration
        config_manager.update_training_config({"batch_size": 128})
        
        # Save configuration
        config_manager.save()
        
        # Create new instance and verify persistence
        new_config_manager = ConfigManager(config_path=setup_test_env / ".config")
        assert new_config_manager.get_training_config()["batch_size"] == 128
    
    def test_environment_override(self, setup_test_env):
        """Test environment variable configuration override."""
        with patch.dict(os.environ, {
            "LLAMA_HOME_BATCH_SIZE": "256",
            "LLAMA_HOME_LEARNING_RATE": "3e-4"
        }):
            config_manager = ConfigManager(config_path=setup_test_env / ".config")
            training_config = config_manager.get_training_config()
            
            assert training_config["batch_size"] == 256
            assert training_config["learning_rate"] == 3e-4
    
    def test_config_inheritance(self, setup_test_env):
        """Test configuration inheritance and merging."""
        # Create base config
        base_config = {
            "training": {
                "base_setting": "value",
                "override_me": "base"
            }
        }
        
        # Create override config
        override_config = {
            "training": {
                "override_me": "override",
                "new_setting": "new"
            }
        }
        
        with patch("builtins.open", mock_open(read_data=toml.dump(base_config))):
            config_manager = ConfigManager(config_path=setup_test_env / ".config")
            config_manager.merge_config(override_config)
            
            training_config = config_manager.get_training_config()
            assert training_config["base_setting"] == "value"
            assert training_config["override_me"] == "override"
            assert training_config["new_setting"] == "new"
    
    def test_config_export(self, setup_test_env):
        """Test configuration export functionality."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        # Export configuration
        export_path = setup_test_env / "exported_config.toml"
        config_manager.export_config(export_path)
        
        # Verify exported file
        assert export_path.exists()
        
        # Load exported config and verify contents
        with open(export_path) as f:
            exported_config = toml.load(f)
            assert exported_config["models"]["llama"]["versions"]["3.3-7b"]["name"] == "Llama 3.3 7B"
    
    def test_config_validation_rules(self, setup_test_env):
        """Test configuration validation rules."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        
        # Test invalid batch size
        with pytest.raises(ValueError):
            config_manager.update_training_config({"batch_size": -1})
        
        # Test invalid learning rate
        with pytest.raises(ValueError):
            config_manager.update_training_config({"learning_rate": 0})
        
        # Test invalid gradient accumulation steps
        with pytest.raises(ValueError):
            config_manager.update_training_config({"gradient_accumulation_steps": 0})
    
    def test_config_defaults(self, setup_test_env):
        """Test configuration defaults handling."""
        # Create minimal config
        minimal_config = {
            "models": {
                "llama": {
                    "versions": {
                        "3.3-7b": {
                            "name": "Llama 3.3 7B"
                        }
                    }
                }
            }
        }
        
        with patch("builtins.open", mock_open(read_data=toml.dump(minimal_config))):
            config_manager = ConfigManager(config_path=setup_test_env / ".config")
            
            # Verify defaults are applied
            model_config = config_manager.get_model_config("llama", "3.3-7b")
            assert "requires_gpu" in model_config
            assert "min_gpu_memory" in model_config
            
            training_config = config_manager.get_training_config()
            assert "batch_size" in training_config
            assert "learning_rate" in training_config 