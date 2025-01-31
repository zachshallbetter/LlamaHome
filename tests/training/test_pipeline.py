"""Tests for training pipeline functionality."""

import os
import pytest
import torch
from typing import Dict, Any
import torch.nn as nn
from unittest.mock import patch, MagicMock

from src.core.evaluation.manager import EvaluationManager
from src.training.pipeline import TrainingPipeline, TrainingConfig
from src.training.data import DataManager
from src.training.monitoring import MonitorManager
from src.training.optimization import Optimizer
from src.training.checkpoint import CheckpointManager
from src.training.metrics import MetricsManager


class TrainingState:
    """Represents the state of model training."""

    def __init__(
        self,
        epoch: int,
        step: int,
        best_metric: float,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
    ):
        self.epoch = epoch
        self.step = step
        self.best_metric = best_metric
        self.model_state = model_state
        self.optimizer_state = optimizer_state

    def update(self, **kwargs):
        """Update state attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        """Convert state to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "best_metric": self.best_metric,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
        }

    @classmethod
    def from_dict(cls, state_dict):
        """Create state from dictionary."""
        return cls(
            epoch=state_dict["epoch"],
            step=state_dict["step"],
            best_metric=state_dict["best_metric"],
            model_state=state_dict["model_state"],
            optimizer_state=state_dict["optimizer_state"],
        )


@pytest.fixture
def mock_config():
    """Create mock pipeline configuration."""
    return {
        "training": {
            "batch_size": 32,
            "num_epochs": 10,
            "gradient_accumulation_steps": 4,
            "eval_steps": 100,
            "save_steps": 500,
            "log_steps": 10,
        },
        "model": {
            "name": "llama",
            "version": "3.3-7b",
            "use_gradient_checkpointing": True,
        },
        "optimization": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "max_grad_norm": 1.0,
        },
        "evaluation": {
            "metrics": ["loss", "perplexity", "accuracy"],
            "num_samples": 1000,
            "generate_samples": True,
        },
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment."""
    # Create necessary directories
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)

    return tmp_path


class TestTrainingPipeline:
    """Test suite for main training pipeline."""

    def test_pipeline_initialization(self, mock_config, setup_test_env):
        """Test pipeline initialization and setup."""
        pipeline = TrainingPipeline(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=TrainingConfig(**mock_config),
        )

        assert pipeline.config.data.batch_size == mock_config["training"]["batch_size"]
        assert pipeline.config.num_epochs == mock_config["training"]["num_epochs"]
        assert pipeline.model.__class__.__name__ == mock_config["model"]["name"]

    def test_training_workflow(self, mock_config, setup_test_env):
        """Test complete training workflow."""
        pipeline = TrainingPipeline(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=TrainingConfig(**mock_config),
        )

        # Create mock model and datasets
        model = nn.Linear(10, 2)  # Simple model for testing
        train_dataset = [
            {"input_ids": torch.randint(0, 1000, (100,)), "label": i % 2}
            for i in range(100)
        ]
        val_dataset = [
            {"input_ids": torch.randint(0, 1000, (100,)), "label": i % 2}
            for i in range(20)
        ]

        # Run training
        with patch.object(pipeline, "_train_epoch") as mock_train:
            with patch.object(pipeline, "_evaluate") as mock_eval:
                pipeline.train(train_data=train_dataset, eval_data=val_dataset)

                assert mock_train.call_count == mock_config["training"]["num_epochs"]
                assert mock_eval.call_count > 0

    def test_training_state(self, mock_config, setup_test_env):
        """Test training state management."""
        state = TrainingState(
            epoch=1, step=100, best_metric=0.85, model_state={}, optimizer_state={}
        )

        # Test state updates
        state.update(step=101, best_metric=0.86)
        assert state.step == 101
        assert state.best_metric == 0.86

        # Test state serialization
        state_dict = state.to_dict()
        new_state = TrainingState.from_dict(state_dict)
        assert new_state.epoch == state.epoch
        assert new_state.best_metric == state.best_metric

    def test_training_hooks(self, mock_config, setup_test_env):
        """Test training hook system."""
        pipeline = TrainingPipeline(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=TrainingConfig(**mock_config),
        )

        # Define test hooks
        pre_training_called = False
        post_epoch_called = False

        def pre_training_hook(state):
            nonlocal pre_training_called
            pre_training_called = True

        def post_epoch_hook(state):
            nonlocal post_epoch_called
            post_epoch_called = True

        # Register hooks
        pipeline.register_hook("pre_training", pre_training_hook)
        pipeline.register_hook("post_epoch", post_epoch_hook)

        # Run training
        model = nn.Linear(10, 2)
        dataset = [{"input_ids": torch.randn(10), "label": 0}]

        pipeline.train(train_data=dataset)

        assert pre_training_called
        assert post_epoch_called


class TestEvaluationManager:
    """Test suite for evaluation functionality."""

    def test_metric_computation(self, mock_config):
        """Test metric computation."""
        evaluator = EvaluationManager(config=mock_config)

        # Create predictions and targets
        predictions = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        targets = torch.tensor([0, 1])

        # Compute metrics
        metrics = evaluator.compute_metrics(predictions, targets)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_generation_evaluation(self, mock_config):
        """Test generation-based evaluation."""
        evaluator = EvaluationManager(config=mock_config)

        # Mock model and tokenizer
        model = MagicMock()
        model.generate.return_value = torch.tensor([[1, 2, 3]])

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Generated text"

        # Test generation evaluation
        samples = evaluator.evaluate_generation(
            model=model, tokenizer=tokenizer, prompts=["Test prompt"]
        )

        assert len(samples) > 0
        assert isinstance(samples[0], str)

    def test_evaluation_logging(self, mock_config, setup_test_env):
        """Test evaluation logging functionality."""
        evaluator = EvaluationManager(config=mock_config)

        # Log some metrics
        metrics = {"loss": 0.5, "accuracy": 0.85, "perplexity": 1.5}

        log_dir = setup_test_env / "logs"
        evaluator.log_metrics(metrics, step=100, log_dir=log_dir)

        # Verify log file exists
        assert (log_dir / "metrics.json").exists()


class TestCheckpointManager:
    """Test suite for checkpoint management."""

    def test_checkpoint_saving(self, mock_config, setup_test_env):
        """Test checkpoint saving functionality."""
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=setup_test_env / "checkpoints", config=mock_config
        )

        # Create mock training state
        state = TrainingState(
            epoch=1,
            step=100,
            best_metric=0.85,
            model_state={"weights": torch.randn(10, 10)},
            optimizer_state={"step": 100},
        )

        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(state)
        assert checkpoint_path.exists()

    def test_checkpoint_loading(self, mock_config, setup_test_env):
        """Test checkpoint loading functionality."""
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=setup_test_env / "checkpoints", config=mock_config
        )

        # Create and save state
        state = TrainingState(
            epoch=1,
            step=100,
            best_metric=0.85,
            model_state={"weights": torch.randn(10, 10)},
            optimizer_state={"step": 100},
        )

        checkpoint_path = checkpoint_manager.save_checkpoint(state)

        # Load state
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        assert loaded_state.epoch == state.epoch
        assert loaded_state.best_metric == state.best_metric

    def test_checkpoint_cleanup(self, mock_config, setup_test_env):
        """Test checkpoint cleanup functionality."""
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=setup_test_env / "checkpoints", config=mock_config
        )

        # Create multiple checkpoints
        for i in range(5):
            state = TrainingState(
                epoch=i,
                step=i * 100,
                best_metric=0.8 + i * 0.01,
                model_state={},
                optimizer_state={},
            )
            checkpoint_manager.save_checkpoint(state)

        # Cleanup old checkpoints
        checkpoint_manager.cleanup(keep_best_n=2)

        # Verify only best checkpoints remain
        remaining = list(checkpoint_manager.checkpoint_dir.glob("*.pt"))
        assert len(remaining) == 2


class TestMetricsManager:
    """Test suite for metrics management."""

    def test_metrics_tracking(self, mock_config):
        """Test metrics tracking functionality."""
        metrics_manager = MetricsManager(config=mock_config)

        # Log metrics
        metrics_manager.update({"train_loss": 0.5, "val_loss": 0.4, "accuracy": 0.85})

        # Get metrics history
        history = metrics_manager.get_history()
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1

    def test_best_metrics(self, mock_config):
        """Test best metrics tracking."""
        metrics_manager = MetricsManager(config=mock_config)

        # Log multiple values
        metrics = [
            {"val_loss": 0.5, "accuracy": 0.8},
            {"val_loss": 0.4, "accuracy": 0.85},
            {"val_loss": 0.45, "accuracy": 0.82},
        ]

        for m in metrics:
            metrics_manager.update(m)

        best = metrics_manager.get_best_metrics()
        assert best["val_loss"] == 0.4
        assert best["accuracy"] == 0.85

    def test_metrics_visualization(self, mock_config, setup_test_env):
        """Test metrics visualization functionality."""
        metrics_manager = MetricsManager(config=mock_config)

        # Log metrics over time
        for i in range(10):
            metrics_manager.update(
                {"train_loss": 1.0 - i * 0.1, "val_loss": 0.9 - i * 0.08}
            )

        # Generate visualization
        output_dir = setup_test_env / "outputs"
        metrics_manager.plot_metrics(output_dir=output_dir)

        assert (output_dir / "metrics.png").exists()
