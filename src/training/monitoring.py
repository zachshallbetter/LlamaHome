"""
Training monitoring and metrics system.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ..core.utils import LogManager, LogTemplates
from ..core.utils.memory_tracker import MemoryTracker

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass
class MonitorConfig:
    """Monitoring configuration."""

    tensorboard: bool = True
    progress_bars: bool = True
    resource_monitoring: bool = True
    log_interval: int = 10
    plot_metrics: bool = True
    track_memory: bool = True
    track_gradients: bool = True
    track_weights: bool = True
    track_attention: bool = True
    tensorboard_dir: str = "runs"


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    track_gradients: bool = True
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 500
    metrics: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["loss", "accuracy", "perplexity"]


class TrainingMetrics:
    """Tracks and manages training metrics."""

    def __init__(self, config: MetricsConfig):
        """Initialize metrics tracker.
        
        Args:
            config: Metrics configuration
        """
        self.config = config
        self.metrics_history = {}
        self.current_step = 0

    def track_gradients(self, model: torch.nn.Module, step: int):
        """Track model gradients.
        
        Args:
            model: Model to track gradients for
            step: Current training step
        """
        if not self.config.track_gradients:
            return

        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        self.update_metric("gradient_norm", grad_norm, step)

    def update_metric(self, name: str, value: float, step: int):
        """Update a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append((step, value))

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all tracked metrics.
        
        Returns:
            List of metric dictionaries
        """
        results = []
        for step in sorted(set(s for m in self.metrics_history.values() for s, _ in m)):
            metrics = {}
            for name, history in self.metrics_history.items():
                values = [v for s, v in history if s == step]
                if values:
                    metrics[name] = values[0]
            results.append({"step": step, "metrics": metrics})
        return results


class MetricsCollector:
    """Collects and manages training metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {}
        self.history = {}
        self.custom_metrics = {}

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Metric value must be numeric, got {type(value)}")
        
        if step is not None and step < 0:
            raise ValueError(f"Step must be non-negative, got {step}")

        self.metrics[name] = value
        if name not in self.history:
            self.history[name] = []
        if step is not None:
            self.history[name].append((step, value))
        else:
            self.history[name].append(value)

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics.
        
        Returns:
            Dictionary of current metric values
        """
        metrics = self.metrics.copy()
        for name, fn in self.custom_metrics.items():
            if name not in metrics and "loss" in metrics:
                metrics[name] = fn(metrics["loss"])
        return metrics

    def get_metric_history(self, name: str) -> List[float]:
        """Get history for a specific metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of historical values
        """
        return [v for _, v in self.history.get(name, [])]

    def register_custom_metric(self, name: str, fn):
        """Register a custom metric function.
        
        Args:
            name: Metric name
            fn: Function to compute metric
        """
        self.custom_metrics[name] = fn

    def save(self, path: Path):
        """Save metrics to file.
        
        Args:
            path: Path to save to
        """
        torch.save({
            "metrics": self.metrics,
            "history": self.history,
        }, path)

    def load(self, path: Path):
        """Load metrics from file.
        
        Args:
            path: Path to load from
        """
        data = torch.load(path)
        self.metrics = data["metrics"]
        self.history = data["history"]


class PerformanceMonitor:
    """Monitors training performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.stats = {}
        self._start_times = {}

    def track(self, name: str):
        """Context manager for tracking performance.
        
        Args:
            name: Name of the operation to track
        """
        class TrackingContext:
            def __init__(self, monitor, name):
                self.monitor = monitor
                self.name = name

            def __enter__(self):
                self.monitor._start_tracking(self.name)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.monitor._end_tracking(self.name)

        return TrackingContext(self, name)

    def _start_tracking(self, name: str):
        """Start tracking an operation.
        
        Args:
            name: Operation name
        """
        self._start_times[name] = torch.cuda.Event(enable_timing=True)
        self._start_times[name].record()

    def _end_tracking(self, name: str):
        """End tracking an operation.
        
        Args:
            name: Operation name
        """
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()

        duration = self._start_times[name].elapsed_time(end_event)
        if name not in self.stats:
            self.stats[name] = {}
        self.stats[name]["duration"] = duration
        self.stats[name]["memory_allocated"] = torch.cuda.memory_allocated()
        if torch.cuda.is_available():
            self.stats[name]["memory_cached"] = torch.cuda.memory_reserved()

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics.
        
        Args:
            name: Optional operation name
            
        Returns:
            Dictionary of statistics
        """
        if name is not None:
            return self.stats.get(name, {})
        return self.stats


class TrainingMetrics:
    """Advanced training metrics collection and visualization."""

    def __init__(
        self, config: Optional[MonitorConfig] = None, model_name: str = "model"
    ):
        """Initialize training metrics.

        Args:
            config: Optional metrics configuration
            model_name: Name of the model
        """
        self.config = config or MonitorConfig()
        self.model_name = model_name
        self._setup_tracking()

        if not PLOTLY_AVAILABLE and self.config.plot_metrics:
            logger.warning("plotly not available, plotting will be disabled")
            self.config.plot_metrics = False

    def _setup_tracking(self) -> None:
        """Set up metrics tracking systems."""
        # TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=f"{self.config.tensorboard_dir}/{self.model_name}"
        )

        # Metrics storage
        self.metrics_history: dict[str, list[float]] = {
            "loss": [],
            "learning_rate": [],
            "memory_usage": [],
            "throughput": [],
            "gradient_norm": [],
            "attention_stats": [],
        }

        # Progress tracking
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

    async def update_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Update training metrics.

        Args:
            step: Current training step
            metrics: Metrics to update
            model: Optional model for additional metrics
        """
        # Basic metrics
        for name, value in metrics.items():
            self.metrics_history[name].append(value)
            self.writer.add_scalar(f"training/{name}", value, step)

        # Memory tracking
        if self.config.track_memory and torch.cuda.is_available():
            memory_metrics = self._track_memory_usage()
            for name, value in memory_metrics.items():
                self.writer.add_scalar(f"memory/{name}", value, step)

        # Model tracking
        if model is not None:
            if self.config.track_gradients:
                await self._track_gradients(model, step)

            if self.config.track_weights:
                await self._track_weights(model, step)

            if self.config.track_attention:
                await self._track_attention(model, step)

    async def _track_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Track gradient statistics.

        Args:
            model: Model to track
            step: Current training step
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{name}", param.grad.data.cpu().numpy(), step
                )

                grad_norm = torch.norm(param.grad.data)
                self.writer.add_scalar(f"gradient_norms/{name}", grad_norm.item(), step)

    async def _track_weights(self, model: torch.nn.Module, step: int) -> None:
        """Track model weight statistics.

        Args:
            model: Model to track
            step: Current training step
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.data.cpu().numpy(), step)

    async def _track_attention(self, model: torch.nn.Module, step: int) -> None:
        """Track attention statistics.

        Args:
            model: Model to track
            step: Current training step
        """
        if hasattr(model, "get_attention_maps"):
            attention_maps = model.get_attention_maps()
            for layer_idx, attention_map in enumerate(attention_maps):
                self.writer.add_image(
                    f"attention/layer_{layer_idx}", attention_map.unsqueeze(0), step
                )

    def _track_memory_usage(self) -> Dict[str, float]:
        """Track memory usage statistics.

        Returns:
            Dict containing memory statistics
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def plot_metrics(self, save_dir: Optional[Path] = None) -> None:
        """Generate interactive training visualizations.

        Args:
            save_dir: Optional directory to save plots
        """
        if not self.config.plot_metrics or not PLOTLY_AVAILABLE:
            return

        # Create figures directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Plot loss curve
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=self.metrics_history["loss"], mode="lines", name="Training Loss"
            )
        )
        fig.update_layout(title="Training Loss Over Time")
        if save_dir:
            fig.write_html(save_dir / "loss_curve.html")

        # Plot learning rate
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=self.metrics_history["learning_rate"],
                mode="lines",
                name="Learning Rate",
            )
        )
        fig.update_layout(title="Learning Rate Schedule")
        if save_dir:
            fig.write_html(save_dir / "learning_rate.html")

        # Plot memory usage
        if self.metrics_history["memory_usage"]:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=self.metrics_history["memory_usage"],
                    mode="lines",
                    name="Memory Usage (GB)",
                )
            )
            fig.update_layout(title="GPU Memory Usage Over Time")
            if save_dir:
                fig.write_html(save_dir / "memory_usage.html")

    def save_metrics(self, save_path: Path) -> None:
        """Save metrics history to file.

        Args:
            save_path: Path to save metrics
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def load_metrics(self, load_path: Path) -> None:
        """Load metrics history from file.

        Args:
            load_path: Path to load metrics from
        """
        load_path = Path(load_path)

        with load_path.open("r") as f:
            self.metrics_history = json.load(f)

    def __enter__(self) -> "TrainingMetrics":
        """Start progress tracking."""
        self.progress.start()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object | None,
    ) -> None:
        """Clean up progress tracking."""
        self.progress.stop()
        self.writer.close()


class DistributedMetrics(TrainingMetrics):
    """Metrics collection for distributed training."""

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        model_name: str = "model",
        world_size: int = 1,
        rank: int = 0,
    ):
        """Initialize distributed metrics.

        Args:
            config: Optional metrics configuration
            model_name: Name of the model
            world_size: Number of distributed processes
            rank: Process rank
        """
        super().__init__(config, f"{model_name}_rank_{rank}")
        self.world_size = world_size
        self.rank = rank

    async def update_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Update metrics with distributed training support.

        Args:
            step: Current training step
            metrics: Metrics to update
            model: Optional model for additional metrics
        """
        # Gather metrics from all processes
        gathered_metrics = {}
        for name, value in metrics.items():
            if torch.distributed.is_initialized():
                gathered_values = [
                    torch.zeros_like(torch.tensor(value))
                    for _ in range(self.world_size)
                ]
                torch.distributed.all_gather(gathered_values, torch.tensor(value))
                gathered_metrics[name] = torch.mean(torch.stack(gathered_values)).item()
            else:
                gathered_metrics[name] = value

        # Update metrics on main process
        if self.rank == 0:
            await super().update_metrics(step, gathered_metrics, model)


class MetricsCallback:
    """Training callback for metrics collection."""

    def __init__(self, metrics: Union[TrainingMetrics, DistributedMetrics]):
        """Initialize metrics callback.

        Args:
            metrics: Metrics collector instance
        """
        self.metrics = metrics

    async def on_batch_end(
        self, step: int, metrics: Dict[str, float], model: torch.nn.Module
    ) -> None:
        """Update metrics after each batch.

        Args:
            step: Current training step
            metrics: Batch metrics
            model: Training model
        """
        await self.metrics.update_metrics(step, metrics, model)

    async def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Generate visualizations after each epoch.

        Args:
            epoch: Current epoch
            metrics: Epoch metrics
        """
        if self.metrics.config.plot_metrics:
            self.metrics.plot_metrics(save_dir=Path(f"figures/epoch_{epoch}"))

        self.metrics.save_metrics(save_path=Path(f"metrics/epoch_{epoch}.json"))


class Monitor:
    """Base monitoring class."""

    async def update(
        self,
        step: int,
        metrics: dict[str, float],
        model: torch.nn.Module | None = None,
    ) -> None:
        """Update monitor with new metrics."""
        raise NotImplementedError


class ProgressMonitor(Monitor):
    """Progress bar monitoring."""

    def __init__(self) -> None:
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

    async def update(
        self,
        step: int,
        metrics: dict[str, float],
        model: torch.nn.Module | None = None,
    ) -> None:
        """Update progress bars."""
        self.progress.update(step, advance=1)


class ResourceMonitor(Monitor):
    """Resource usage monitoring."""

    def __init__(self) -> None:
        self.memory_tracker = MemoryTracker()

    async def update(
        self,
        step: int,
        metrics: dict[str, float],
        model: torch.nn.Module | None = None,
    ) -> None:
        """Update resource metrics."""
        if torch.cuda.is_available():
            metrics["gpu_memory"] = torch.cuda.memory_allocated() / 1024**3
            metrics["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1024**3


class TensorboardMonitor(Monitor):
    """Tensorboard monitoring."""

    def __init__(self, model_name: str) -> None:
        self.writer = SummaryWriter(f"runs/{model_name}")

    async def update(
        self,
        step: int,
        metrics: dict[str, float],
        model: torch.nn.Module | None = None,
    ) -> None:
        """Update tensorboard metrics."""
        for name, value in metrics.items():
            self.writer.add_scalar(f"training/{name}", value, step)


class MonitorManager:
    """Manages multiple monitoring components."""

    def __init__(self, config: MonitorConfig, model_name: str) -> None:
        self.config = config
        self.model_name = model_name
        self.monitors: dict[str, Monitor] = {}
        self._setup_monitors()

    def _setup_monitors(self) -> None:
        """Set up monitoring components."""
        if self.config.progress_bars:
            self.monitors["progress"] = ProgressMonitor()
        if self.config.resource_monitoring:
            self.monitors["resource"] = ResourceMonitor()
        if self.config.tensorboard:
            self.monitors["tensorboard"] = TensorboardMonitor(self.model_name)

    async def update_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        model: torch.nn.Module | None = None,
    ) -> None:
        """Update monitoring metrics."""
        for monitor in self.monitors.values():
            await monitor.update(step, metrics, model)


class MonitorError(Exception):
    """Monitoring error."""

    pass
