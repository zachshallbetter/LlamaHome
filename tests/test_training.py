"""Training pipeline tests."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import toml
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.core.config import ConfigManager
from src.training.pipeline import TrainingPipeline
from src.training.optimization import Optimizer
from src.training.monitoring import MetricsCollector
from src.training.distributed import DistributedTrainer


@pytest.fixture
def mock_config() -> dict[str, float]:
    """Create mock training configuration."""
    return {
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "optimizer": {
                "name": "adamw",
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
            },
            "scheduler": {"name": "cosine", "num_cycles": 1},
        },
        "distributed": {
            "backend": "nccl",
            "world_size": 1,
            "init_method": "tcp://localhost:23456",
        },
    }


@pytest.fixture
def setup_test_env(tmp_path: Path, mock_config: dict[str, float]) -> Path:
    """Set up test environment with configuration."""
    # Create config directory
    config_dir = tmp_path / ".config"
    config_dir.mkdir(parents=True)

    # Create config file
    config_file = config_dir / "training_config.toml"
    with open(config_file, "w") as f:
        toml.dump(mock_config, f)

    # Create data directories
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # Create sample training file
    train_file = data_dir / "train.jsonl"
    with open(train_file, "w") as f:
        f.write('{"text": "Sample training text 1"}\n')
        f.write('{"text": "Sample training text 2"}\n')

    return tmp_path


class TestTrainingPipeline:
    """Test suite for training pipeline."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization."""
        config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10}
        pipeline = TrainingPipeline(config)
        assert pipeline.config == config
        assert pipeline.current_epoch == 0

    def test_data_loading(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test training data loading and preprocessing."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        # Test data loading
        data_path = setup_test_env / "data" / "train.jsonl"
        dataset = pipeline.load_dataset(data_path)

        assert len(dataset) > 0
        assert "text" in dataset[0]

    def test_optimizer_setup(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test optimizer and scheduler setup."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        with patch("torch.optim.AdamW") as mock_adamw:
            optimizer = pipeline.setup_optimizer()
            mock_adamw.assert_called_once()

            # Test scheduler setup
            scheduler = pipeline.setup_scheduler(optimizer)
            assert scheduler is not None

    def test_training_step(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test single training step."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        # Mock batch data
        batch = {
            "input_ids": torch.randint(0, 32000, (32, 128)),
            "attention_mask": torch.ones(32, 128),
            "labels": torch.randint(0, 32000, (32, 128)),
        }

        with patch.object(pipeline.model, "forward") as mock_forward:
            mock_forward.return_value.loss = torch.tensor(0.5)

            loss = pipeline.training_step(batch)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() > 0

    def test_validation_step(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test validation step."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        # Mock batch data
        batch = {
            "input_ids": torch.randint(0, 32000, (32, 128)),
            "attention_mask": torch.ones(32, 128),
            "labels": torch.randint(0, 32000, (32, 128)),
        }

        with patch.object(pipeline.model, "forward") as mock_forward:
            mock_forward.return_value.loss = torch.tensor(0.3)

            metrics = pipeline.validation_step(batch)
            assert "val_loss" in metrics
            assert metrics["val_loss"] > 0

    @pytest.mark.integration
    def test_full_training_loop(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test complete training loop."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        # Mock dataset
        train_dataset = MagicMock()
        train_dataset.__len__.return_value = 100

        val_dataset = MagicMock()
        val_dataset.__len__.return_value = 20

        with patch.object(pipeline, "training_step") as mock_train_step:
            with patch.object(pipeline, "validation_step") as mock_val_step:
                mock_train_step.return_value = torch.tensor(0.5)
                mock_val_step.return_value = {"val_loss": 0.3}

                history = pipeline.train(
                    train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=1
                )

                assert "train_loss" in history
                assert "val_loss" in history

    def test_distributed_training(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test distributed training setup."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")

        with patch("torch.distributed.init_process_group") as mock_init_process:
            trainer = DistributedTrainer(
                model_name="llama",
                model_version="3.3-7b",
                config_manager=config_manager,
            )

            mock_init_process.assert_called_once()
            assert trainer.world_size == mock_config["distributed"]["world_size"]

    def test_metrics_collection(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test training metrics collection."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        metrics_collector = MetricsCollector()

        # Test metrics logging
        metrics_collector.log_metric("train_loss", 0.5)
        metrics_collector.log_metric("val_loss", 0.3)

        # Test metrics retrieval
        metrics = metrics_collector.get_metrics()
        assert "train_loss" in metrics
        assert "val_loss" in metrics

    def test_optimization_features(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test training optimization features."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        optimizer = Optimizer(config_manager)

        # Test gradient accumulation
        assert optimizer.gradient_accumulation_steps == 4

        # Test gradient clipping
        assert optimizer.max_grad_norm == 1.0

        # Test learning rate scheduling
        assert optimizer.get_scheduler_name() == "cosine"

    def test_checkpointing(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test model checkpointing."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        # Test checkpoint saving
        checkpoint_path = setup_test_env / "checkpoints" / "checkpoint.pt"
        pipeline.save_checkpoint(
            checkpoint_path,
            epoch=1,
            step=1000,
            optimizer_state={"state": "test"},
            metrics={"loss": 0.5},
        )

        # Test checkpoint loading
        loaded_state = pipeline.load_checkpoint(checkpoint_path)
        assert loaded_state["epoch"] == 1
        assert loaded_state["step"] == 1000
        assert loaded_state["metrics"]["loss"] == 0.5

    def test_error_handling(self, setup_test_env: Path, mock_config: dict[str, float]) -> None:
        """Test training error handling."""
        config_manager = ConfigManager(config_path=setup_test_env / ".config")
        pipeline = TrainingPipeline(
            model_name="llama", model_version="3.3-7b", config_manager=config_manager
        )

        # Test invalid dataset
        with pytest.raises(ValueError):
            pipeline.load_dataset("nonexistent.jsonl")

        # Test invalid checkpoint
        with pytest.raises(FileNotFoundError):
            pipeline.load_checkpoint("nonexistent.pt")

        # Test invalid configuration
        with pytest.raises(ValueError):
            pipeline.update_config({"invalid_key": "value"})
