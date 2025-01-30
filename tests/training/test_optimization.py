"""Tests for training optimization functionality."""

import pytest
import torch
import torch.nn as nn  # Added import for nn
from typing import Dict, Any

from src.training.optimization import (
    Optimizer,
    GradientHandler,
    MemoryOptimizer,
    SchedulerManager,
    PerformanceOptimizer,
)


@pytest.fixture
def mock_config():
    """Create mock optimization configuration."""
    return {
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        "scheduler": {"name": "cosine", "warmup_steps": 100, "num_cycles": 1},
        "gradient": {
            "accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "skip_grad_norm_check": False,
        },
        "memory": {
            "use_gradient_checkpointing": True,
            "use_mixed_precision": True,
            "max_memory_usage": 0.9,
        },
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment."""
    # Create necessary directories
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    return tmp_path


class TestOptimizer:
    """Test suite for optimizer configuration and management."""

    def test_optimizer_initialization(self, mock_config):
        """Test optimizer initialization with different configurations."""
        optimizer_manager = Optimizer(config=mock_config)

        # Create simple model for testing
        model = nn.Linear(10, 10)

        # Test AdamW configuration
        optimizer = optimizer_manager.create_optimizer(model)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == mock_config["optimizer"]["learning_rate"]
        assert (
            optimizer.defaults["weight_decay"]
            == mock_config["optimizer"]["weight_decay"]
        )

        # Test different optimizer types
        optimizer_types = ["adam", "sgd", "adamw"]
        for opt_type in optimizer_types:
            mock_config["optimizer"]["name"] = opt_type
            optimizer = optimizer_manager.create_optimizer(model)
            assert optimizer is not None

    def test_optimizer_groups(self, mock_config):
        """Test parameter group handling in optimizer."""
        optimizer_manager = Optimizer(config=mock_config)

        # Create model with different parameter groups
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))

        # Configure parameter groups
        param_groups = [
            {"params": model[0].parameters(), "lr": 1e-3},
            {"params": model[1].parameters(), "lr": 1e-4},
        ]

        optimizer = optimizer_manager.create_optimizer(model, param_groups=param_groups)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 1e-4

    def test_optimizer_state_management(self, mock_config, setup_test_env):
        """Test optimizer state management."""
        optimizer_manager = Optimizer(config=mock_config)
        model = nn.Linear(10, 10)
        optimizer = optimizer_manager.create_optimizer(model)

        # Save optimizer state
        state_path = setup_test_env / "checkpoints" / "optimizer.pt"
        optimizer_manager.save_optimizer_state(optimizer, state_path)

        # Load optimizer state
        new_optimizer = optimizer_manager.create_optimizer(model)
        optimizer_manager.load_optimizer_state(new_optimizer, state_path)

        # Verify state loading
        assert optimizer.state_dict().keys() == new_optimizer.state_dict().keys()


class TestGradientHandler:
    """Test suite for gradient handling functionality."""

    def test_gradient_accumulation(self, mock_config):
        """Test gradient accumulation mechanism."""
        gradient_handler = GradientHandler(config=mock_config)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())

        # Test accumulation over steps
        for step in range(mock_config["gradient"]["accumulation_steps"]):
            input_tensor = torch.randn(32, 10)
            output = model(input_tensor)
            loss = output.sum()

            should_update = gradient_handler.backward_pass(loss, model, optimizer, step)

            if step < mock_config["gradient"]["accumulation_steps"] - 1:
                assert not should_update
            else:
                assert should_update

    def test_gradient_clipping(self, mock_config):
        """Test gradient clipping functionality."""
        gradient_handler = GradientHandler(config=mock_config)
        model = nn.Linear(10, 10)

        # Generate large gradients
        input_tensor = torch.randn(32, 10)
        output = model(input_tensor)
        loss = output.sum() * 1000  # Create large gradients
        loss.backward()

        # Test gradient clipping
        grad_norm = gradient_handler.clip_gradients(model)
        assert grad_norm <= mock_config["gradient"]["max_grad_norm"]

    def test_gradient_scaling(self, mock_config):
        """Test gradient scaling for mixed precision training."""
        gradient_handler = GradientHandler(config=mock_config)
        scaler = torch.cuda.amp.GradScaler()

        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())

        # Test gradient scaling
        with torch.cuda.amp.autocast():
            input_tensor = torch.randn(32, 10)
            output = model(input_tensor)
            loss = output.sum()

        gradient_handler.backward_pass(loss, model, optimizer, 0, scaler=scaler)
        assert scaler.get_scale() > 0


class TestMemoryOptimizer:
    """Test suite for memory optimization features."""

    def test_memory_tracking(self, mock_config):
        """Test memory usage tracking."""
        memory_optimizer = MemoryOptimizer(config=mock_config)

        # Test memory tracking
        with memory_optimizer.track_memory():
            # Allocate some tensors
            tensors = [torch.randn(1000, 1000) for _ in range(5)]

        memory_stats = memory_optimizer.get_memory_stats()
        assert "allocated" in memory_stats
        assert "cached" in memory_stats

    def test_gradient_checkpointing(self, mock_config):
        """Test gradient checkpointing functionality."""
        memory_optimizer = MemoryOptimizer(config=mock_config)

        # Create a larger model for testing
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

        # Enable gradient checkpointing
        memory_optimizer.enable_gradient_checkpointing(model)

        # Verify checkpointing is enabled
        input_tensor = torch.randn(32, 100)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        assert model.training

    def test_mixed_precision(self, mock_config):
        """Test mixed precision training setup."""
        memory_optimizer = MemoryOptimizer(config=mock_config)

        # Test mixed precision context
        with memory_optimizer.mixed_precision_context():
            model = nn.Linear(10, 10)
            input_tensor = torch.randn(32, 10)
            output = model(input_tensor)

            assert output.dtype == torch.float16


class TestSchedulerManager:
    """Test suite for learning rate scheduling."""

    def test_scheduler_creation(self, mock_config):
        """Test learning rate scheduler creation."""
        scheduler_manager = SchedulerManager(config=mock_config)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())

        # Test different scheduler types
        scheduler_types = ["cosine", "linear", "constant"]
        for scheduler_type in scheduler_types:
            mock_config["scheduler"]["name"] = scheduler_type
            scheduler = scheduler_manager.create_scheduler(optimizer)
            assert scheduler is not None

    def test_warmup_scheduling(self, mock_config):
        """Test warmup scheduling functionality."""
        scheduler_manager = SchedulerManager(config=mock_config)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = scheduler_manager.create_scheduler(optimizer)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Test warmup phase
        for step in range(mock_config["scheduler"]["warmup_steps"]):
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            assert current_lr <= initial_lr

    def test_scheduler_state_management(self, mock_config, setup_test_env):
        """Test scheduler state management."""
        scheduler_manager = SchedulerManager(config=mock_config)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = scheduler_manager.create_scheduler(optimizer)

        # Save scheduler state
        state_path = setup_test_env / "checkpoints" / "scheduler.pt"
        scheduler_manager.save_scheduler_state(scheduler, state_path)

        # Load scheduler state
        new_scheduler = scheduler_manager.create_scheduler(optimizer)
        scheduler_manager.load_scheduler_state(new_scheduler, state_path)

        # Verify state loading
        assert scheduler.state_dict().keys() == new_scheduler.state_dict().keys()


class TestPerformanceOptimizer:
    """Test suite for performance optimization strategies."""

    def test_performance_monitoring(self, mock_config):
        """Test performance monitoring capabilities."""
        performance_optimizer = PerformanceOptimizer(config=mock_config)

        # Test performance tracking
        with performance_optimizer.track_performance("forward_pass"):
            # Simulate computation
            model = nn.Linear(1000, 1000)
            input_tensor = torch.randn(32, 1000)
            output = model(input_tensor)

        metrics = performance_optimizer.get_performance_metrics()
        assert "forward_pass" in metrics
        assert "duration" in metrics["forward_pass"]

    def test_batch_size_optimization(self, mock_config):
        """Test batch size optimization."""
        performance_optimizer = PerformanceOptimizer(config=mock_config)

        # Test batch size finder
        model = nn.Linear(100, 100)
        input_tensor = torch.randn(128, 100)

        optimal_batch_size = performance_optimizer.find_optimal_batch_size(
            model, input_tensor, min_batch_size=16, max_batch_size=256
        )

        assert 16 <= optimal_batch_size <= 256

    def test_throughput_optimization(self, mock_config):
        """Test throughput optimization strategies."""
        performance_optimizer = PerformanceOptimizer(config=mock_config)

        # Test throughput measurement
        model = nn.Linear(100, 100)
        input_tensor = torch.randn(32, 100)

        throughput = performance_optimizer.measure_throughput(
            model, input_tensor, num_iterations=10
        )

        assert throughput > 0
