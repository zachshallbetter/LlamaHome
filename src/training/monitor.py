"""Training monitoring functionality."""

import logging
from typing import Dict, Any, Optional, List
import torch
from pathlib import Path

from .metrics import MetricsConfig, TrainingMetrics

logger = logging.getLogger(__name__)


class MonitorManager:
    """Manages training monitoring components."""

    def __init__(
        self,
        config: MetricsConfig,
        model_name: str,
        output_dir: Optional[Path] = None
    ):
        """Initialize monitor manager.
        
        Args:
            config: Metrics configuration
            model_name: Name of the model being monitored
            output_dir: Optional output directory for logs and artifacts
        """
        self.config = config
        self.model_name = model_name
        self.output_dir = output_dir or Path("outputs")
        self.metrics = TrainingMetrics(config)
        self._setup_monitoring()

    def _setup_monitoring(self) -> None:
        """Set up monitoring components."""
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components based on config
        if self.config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=str(self.output_dir / self.config.tensorboard_dir)
            )

    def update_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        model: Optional[torch.nn.Module] = None
    ) -> None:
        """Update monitoring metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Current training step
            model: Optional model for additional metrics
        """
        # Update basic metrics
        for name, value in metrics.items():
            self.metrics.update_metric(name, value, step)
            if hasattr(self, "writer"):
                self.writer.add_scalar(f"training/{name}", value, step)

        # Track model metrics if provided
        if model is not None and self.config.track_gradients:
            self.metrics.track_gradients(model, step)

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get current metrics.
        
        Returns:
            Dictionary of metric histories
        """
        return {
            name: [v for _, v in history]
            for name, history in self.metrics.metrics_history.items()
        }

    def save_state(self, path: Optional[Path] = None) -> None:
        """Save monitor state.
        
        Args:
            path: Optional path to save to
        """
        if path is None:
            path = self.output_dir / "monitor_state.pt"
        self.metrics.save_state(str(path))

    def load_state(self, path: Path) -> None:
        """Load monitor state.
        
        Args:
            path: Path to load from
        """
        self.metrics.load_state(str(path))

    def close(self) -> None:
        """Clean up monitoring resources."""
        if hasattr(self, "writer"):
            self.writer.close() 