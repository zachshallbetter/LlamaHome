"""Tests for checkpoint management functionality."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.checkpoint import CheckpointManager, CheckpointConfig


@pytest.fixture
def mock_config():
    """Create mock checkpoint configuration."""
    return {
        "checkpoint": {
            "save_dir": "checkpoints",
            "save_steps": 1000,
            "save_epochs": 1,
            "keep_last_n": 3,
            "save_best": True,
            "metric_for_best": "loss",
            "greater_is_better": False,
        }
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment."""
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    return tmp_path


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)


class TestCheckpointManager:
    """Test suite for checkpoint management."""

    def test_initialization(self, mock_config, setup_test_env):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(mock_config, "test_model")
        assert isinstance(manager.config, CheckpointConfig)
        assert manager.model_name == "test_model"
        assert manager.save_dir.exists()

    def test_save_load_checkpoint(self, mock_config, setup_test_env):
        """Test saving and loading checkpoints."""
        manager = CheckpointManager(mock_config, "test_model")
        
        # Create test model and optimizer
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        metrics = {"loss": 0.5, "accuracy": 0.95}
        checkpoint_path = manager.save(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            step=1000,
            epoch=1
        )
        
        assert checkpoint_path.exists()
        assert (checkpoint_path / "model.pt").exists()
        assert (checkpoint_path / "training_state.pt").exists()
        
        # Load checkpoint
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        state = manager.load(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer
        )
        
        assert state["step"] == 1000
        assert state["epoch"] == 1
        assert state["metrics"] == metrics

    def test_best_checkpoint_tracking(self, mock_config, setup_test_env):
        """Test best checkpoint tracking."""
        manager = CheckpointManager(mock_config, "test_model")
        model = SimpleModel()
        
        # Save first checkpoint
        path1 = manager.save(model=model, metrics={"loss": 0.5})
        is_best = manager.update_best(metrics={"loss": 0.5}, checkpoint_path=path1)
        assert is_best
        
        # Save second checkpoint with worse metric
        path2 = manager.save(model=model, metrics={"loss": 0.6})
        is_best = manager.update_best(metrics={"loss": 0.6}, checkpoint_path=path2)
        assert not is_best
        
        # Save third checkpoint with better metric
        path3 = manager.save(model=model, metrics={"loss": 0.4})
        is_best = manager.update_best(metrics={"loss": 0.4}, checkpoint_path=path3)
        assert is_best
        
        # Verify best checkpoint
        best_info = manager.get_best_checkpoint()
        assert best_info is not None
        assert best_info["path"] == str(path3)

    def test_checkpoint_cleanup(self, mock_config, setup_test_env):
        """Test cleanup of old checkpoints."""
        mock_config["checkpoint"]["keep_last_n"] = 2
        manager = CheckpointManager(mock_config, "test_model")
        model = SimpleModel()
        
        # Save multiple checkpoints
        paths = []
        for i in range(5):
            path = manager.save(model=model, step=i)
            paths.append(path)
        
        # Verify only last N checkpoints remain
        existing_paths = list(manager.save_dir.glob("checkpoint_*"))
        assert len(existing_paths) == 2
        
        # Verify the most recent checkpoints are kept
        assert paths[-1].exists()
        assert paths[-2].exists()
        assert not paths[0].exists()

    def test_safetensors_support(self, mock_config, setup_test_env):
        """Test safetensors format support."""
        mock_config["checkpoint"]["save_format"] = "safetensors"
        manager = CheckpointManager(mock_config, "test_model")
        model = SimpleModel()
        
        with patch("safetensors.torch.save_file") as mock_save:
            path = manager.save(model=model)
            mock_save.assert_called_once()
            
        with patch("safetensors.torch.load_file") as mock_load:
            manager.load(path, model=model)
            mock_load.assert_called_once()

    def test_error_handling(self, mock_config, setup_test_env):
        """Test error handling."""
        manager = CheckpointManager(mock_config, "test_model")
        
        # Test loading non-existent checkpoint
        with pytest.raises(ValueError):
            manager.load(Path("nonexistent"))
        
        # Test loading without checkpoints
        with pytest.raises(ValueError):
            manager.load()
        
        # Test updating best with missing metric
        assert not manager.update_best(metrics={}, checkpoint_path=Path("dummy"))
        
        # Test corrupted checkpoint history
        with patch("builtins.open", side_effect=Exception):
            manager._load_checkpoint_history()
            assert manager.checkpoints == [] 