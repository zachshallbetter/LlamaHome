"""Tests for training monitoring functionality."""

import pytest
import torch
from typing import Dict, Any

from src.training.monitoring import TrainingMetrics, MetricsConfig
from src.core.config import ConfigManager


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment with necessary directories."""
    # Create logs directory
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)

    # Create visualizations directory
    viz_dir = tmp_path / "visualizations"
    viz_dir.mkdir(parents=True)

    return tmp_path


class TestMetricsCollector:
    """Test suite for metrics collection system."""

    def test_basic_metrics_logging(self):
        """Test basic metrics logging functionality."""
        collector = MetricsCollector()

        # Log some metrics
        collector.log_metric("train_loss", 0.5)
        collector.log_metric("val_loss", 0.3)
        collector.log_metric("learning_rate", 1e-4)

        # Get logged metrics
        metrics = collector.get_metrics()

        assert "train_loss" in metrics
        assert metrics["train_loss"] == 0.5
        assert "val_loss" in metrics
        assert metrics["val_loss"] == 0.3

    def test_step_based_logging(self):
        """Test step-based metrics logging."""
        collector = MetricsCollector()

        # Log metrics for different steps
        for step in range(5):
            collector.log_metric("loss", 0.5 - step * 0.1, step=step)

        # Get metrics history
        history = collector.get_metric_history("loss")
        assert len(history) == 5
        assert history[0] == 0.5
        assert history[-1] == 0.1

    def test_distributed_metrics(self):
        """Test metrics collection in distributed setting."""
        aggregator = DistributedMetricsAggregator(world_size=2)

        # Simulate metrics from different processes
        metrics_process0 = {"loss": 0.5, "accuracy": 0.8}
        metrics_process1 = {"loss": 0.3, "accuracy": 0.9}

        # Aggregate metrics
        aggregated = aggregator.aggregate([metrics_process0, metrics_process1])

        assert aggregated["loss"] == pytest.approx(0.4)  # Mean of 0.5 and 0.3
        assert aggregated["accuracy"] == pytest.approx(0.85)  # Mean of 0.8 and 0.9

    def test_custom_metrics(self):
        """Test custom metrics computation."""
        collector = MetricsCollector()

        # Define custom metric
        def perplexity(loss):
            return torch.exp(torch.tensor(loss)).item()

        # Register custom metric
        collector.register_custom_metric("perplexity", perplexity)

        # Log loss and compute perplexity
        collector.log_metric("loss", 2.0)
        metrics = collector.get_metrics()

        assert "perplexity" in metrics
        assert metrics["perplexity"] == pytest.approx(7.389, rel=1e-3)

    def test_metric_validation(self):
        """Test metric validation and error handling."""
        collector = MetricsCollector()

        # Test invalid metric value
        with pytest.raises(ValueError):
            collector.log_metric("invalid_loss", "not_a_number")

        # Test invalid step value
        with pytest.raises(ValueError):
            collector.log_metric("loss", 0.5, step=-1)

    def test_metric_serialization(self, setup_test_env):
        """Test metric serialization and loading."""
        collector = MetricsCollector()

        # Log some metrics
        collector.log_metric("train_loss", 0.5)
        collector.log_metric("val_loss", 0.3)

        # Save metrics
        save_path = setup_test_env / "metrics.pt"
        collector.save(save_path)

        # Load metrics in new collector
        new_collector = MetricsCollector()
        new_collector.load(save_path)

        loaded_metrics = new_collector.get_metrics()
        assert loaded_metrics["train_loss"] == 0.5
        assert loaded_metrics["val_loss"] == 0.3

    def test_gradient_tracking(self):
        """Test gradient tracking functionality."""
        metrics = TrainingMetrics(MetricsConfig(track_gradients=True))
        model = torch.nn.Linear(10, 2)

        # Create sample tensors and track them
        tensors = [torch.randn(5, 10) for _ in range(3)]
        metrics.track_gradients(model, step=0)

        assert len(metrics.get_results()) > 0
        assert "gradient_norm" in metrics.get_results()[0].metrics


class TestPerformanceMonitor:
    """Test suite for performance monitoring."""

    def test_basic_monitoring(self):
        """Test basic performance monitoring."""
        monitor = PerformanceMonitor()

        # Start monitoring
        with monitor.track("forward_pass"):
            # Simulate computation
            torch.randn(1000, 1000).mm(torch.randn(1000, 1000))

        stats = monitor.get_stats("forward_pass")
        assert "duration" in stats
        assert "memory_allocated" in stats

    def test_nested_monitoring(self):
        """Test nested performance monitoring."""
        monitor = PerformanceMonitor()

        with monitor.track("outer"):
            torch.randn(100, 100)
            with monitor.track("inner"):
                torch.randn(100, 100)

        stats = monitor.get_stats()
        assert "outer" in stats
        assert "inner" in stats
        assert stats["outer"]["duration"] > stats["inner"]["duration"]

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        monitor = PerformanceMonitor()

        with monitor.track("memory_test"):
            # Allocate some tensors
            tensors = [torch.randn(1000, 1000) for _ in range(5)]

        stats = monitor.get_stats("memory_test")
        assert stats["memory_allocated"] > 0
        assert "memory_peak" in stats

    def test_gpu_monitoring(self):
        """Test GPU monitoring if available."""
        if torch.cuda.is_available():
            monitor = PerformanceMonitor()

            with monitor.track("gpu_test"):
                # Perform GPU operation
                torch.randn(1000, 1000).cuda()

            stats = monitor.get_stats("gpu_test")
            assert "gpu_memory_allocated" in stats
            assert "gpu_memory_cached" in stats


class TestVisualizer:
    """Test suite for visualization system."""

    def test_loss_curve_plotting(self, setup_test_env):
        """Test loss curve visualization."""
        visualizer = Visualizer(output_dir=setup_test_env / "visualizations")

        # Generate sample loss data
        steps = np.arange(100)
        train_loss = np.exp(-steps * 0.01) + np.random.normal(0, 0.1, 100)
        val_loss = np.exp(-steps * 0.01) + np.random.normal(0, 0.1, 100)

        # Plot losses
        visualizer.plot_losses(
            train_loss=train_loss,
            val_loss=val_loss,
            steps=steps,
            title="Training Progress",
        )

        assert (setup_test_env / "visualizations" / "loss_curve.png").exists()

    def test_metric_visualization(self, setup_test_env):
        """Test general metric visualization."""
        visualizer = Visualizer(output_dir=setup_test_env / "visualizations")

        # Generate sample metrics
        steps = np.arange(50)
        metrics = {
            "accuracy": np.linspace(0.5, 0.9, 50) + np.random.normal(0, 0.05, 50),
            "perplexity": np.exp(-steps * 0.02) * 10 + np.random.normal(0, 0.5, 50),
        }

        # Plot metrics
        visualizer.plot_metrics(metrics=metrics, steps=steps, title="Training Metrics")

        assert (setup_test_env / "visualizations" / "metrics.png").exists()

    def test_attention_visualization(self, setup_test_env):
        """Test attention pattern visualization."""
        visualizer = Visualizer(output_dir=setup_test_env / "visualizations")

        # Generate sample attention weights
        attention_weights = torch.softmax(torch.randn(1, 8, 32, 32), dim=-1)

        # Plot attention patterns
        visualizer.plot_attention(
            attention_weights=attention_weights, title="Attention Patterns"
        )

        assert (setup_test_env / "visualizations" / "attention.png").exists()

    def test_interactive_visualizations(self, setup_test_env):
        """Test interactive visualization features."""
        visualizer = Visualizer(output_dir=setup_test_env / "visualizations")

        # Generate sample training history
        history = {
            "train_loss": np.random.normal(1.0, 0.2, 100),
            "val_loss": np.random.normal(0.8, 0.2, 100),
            "learning_rate": np.logspace(-4, -6, 100),
        }

        # Create interactive dashboard
        visualizer.create_dashboard(
            history=history,
            save_path=setup_test_env / "visualizations" / "dashboard.html",
        )

        assert (setup_test_env / "visualizations" / "dashboard.html").exists()

    def test_export_formats(self, setup_test_env):
        """Test different export formats for visualizations."""
        visualizer = Visualizer(output_dir=setup_test_env / "visualizations")

        # Generate sample data
        steps = np.arange(50)
        values = np.random.normal(0, 1, 50)

        # Test different formats
        formats = ["png", "pdf", "svg"]
        for fmt in formats:
            visualizer.plot_metric(
                values=values, steps=steps, title="Test Plot", format=fmt
            )
            assert (setup_test_env / "visualizations" / f"metric.{fmt}").exists()
