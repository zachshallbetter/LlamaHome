"""
Training monitoring and metrics system.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import time

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

@dataclass
class MetricsConfig:
    """Metrics configuration."""
    tensorboard_dir: str = "runs"
    log_interval: int = 10
    save_interval: int = 100
    plot_metrics: bool = True
    track_memory: bool = True
    track_gradients: bool = True
    track_weights: bool = True
    track_attention: bool = True

class TrainingMetrics:
    """Advanced training metrics collection and visualization."""
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        model_name: str = "model"
    ):
        self.config = config or MetricsConfig()
        self.model_name = model_name
        self._setup_tracking()
        
    def _setup_tracking(self) -> None:
        """Set up metrics tracking systems."""
        # TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=f"{self.config.tensorboard_dir}/{self.model_name}"
        )
        
        # Metrics storage
        self.metrics_history = {
            "loss": [],
            "learning_rate": [],
            "memory_usage": [],
            "throughput": [],
            "gradient_norm": [],
            "attention_stats": []
        }
        
        # Progress tracking
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        )
        
    async def update_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None
    ) -> None:
        """Update training metrics."""
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
    
    async def _track_gradients(
        self,
        model: torch.nn.Module,
        step: int
    ) -> None:
        """Track gradient statistics."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{name}",
                    param.grad.data.cpu().numpy(),
                    step
                )
                
                grad_norm = torch.norm(param.grad.data)
                self.writer.add_scalar(
                    f"gradient_norms/{name}",
                    grad_norm.item(),
                    step
                )
    
    async def _track_weights(
        self,
        model: torch.nn.Module,
        step: int
    ) -> None:
        """Track model weight statistics."""
        for name, param in model.named_parameters():
            self.writer.add_histogram(
                f"weights/{name}",
                param.data.cpu().numpy(),
                step
            )
    
    async def _track_attention(
        self,
        model: torch.nn.Module,
        step: int
    ) -> None:
        """Track attention statistics."""
        if hasattr(model, "get_attention_maps"):
            attention_maps = model.get_attention_maps()
            for layer_idx, attention_map in enumerate(attention_maps):
                self.writer.add_image(
                    f"attention/layer_{layer_idx}",
                    attention_map.unsqueeze(0),
                    step
                )
    
    def _track_memory_usage(self) -> Dict[str, float]:
        """Track memory usage statistics."""
        if not torch.cuda.is_available():
            return {}
            
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,      # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
        }
    
    def plot_metrics(self, save_dir: Optional[Path] = None) -> None:
        """Generate interactive training visualizations."""
        if not self.config.plot_metrics:
            return
            
        # Create figures directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot loss curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.metrics_history["loss"],
            mode="lines",
            name="Training Loss"
        ))
        fig.update_layout(title="Training Loss Over Time")
        if save_dir:
            fig.write_html(save_dir / "loss_curve.html")
        
        # Plot learning rate
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.metrics_history["learning_rate"],
            mode="lines",
            name="Learning Rate"
        ))
        fig.update_layout(title="Learning Rate Schedule")
        if save_dir:
            fig.write_html(save_dir / "learning_rate.html")
        
        # Plot memory usage
        if self.metrics_history["memory_usage"]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=self.metrics_history["memory_usage"],
                mode="lines",
                name="Memory Usage (GB)"
            ))
            fig.update_layout(title="GPU Memory Usage Over Time")
            if save_dir:
                fig.write_html(save_dir / "memory_usage.html")
    
    def save_metrics(self, save_path: Path) -> None:
        """Save metrics history to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with save_path.open("w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def load_metrics(self, load_path: Path) -> None:
        """Load metrics history from file."""
        load_path = Path(load_path)
        
        with load_path.open("r") as f:
            self.metrics_history = json.load(f)
    
    def __enter__(self):
        """Start progress tracking."""
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up progress tracking."""
        self.progress.stop()
        self.writer.close()

class DistributedMetrics(TrainingMetrics):
    """Metrics collection for distributed training."""
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        model_name: str = "model",
        world_size: int = 1,
        rank: int = 0
    ):
        super().__init__(config, f"{model_name}_rank_{rank}")
        self.world_size = world_size
        self.rank = rank
    
    async def update_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None
    ) -> None:
        """Update metrics with distributed training support."""
        # Gather metrics from all processes
        gathered_metrics = {}
        for name, value in metrics.items():
            if torch.distributed.is_initialized():
                gathered_values = [torch.zeros_like(torch.tensor(value)) for _ in range(self.world_size)]
                torch.distributed.all_gather(gathered_values, torch.tensor(value))
                gathered_metrics[name] = torch.mean(torch.stack(gathered_values)).item()
            else:
                gathered_metrics[name] = value
        
        # Update metrics on main process
        if self.rank == 0:
            await super().update_metrics(step, gathered_metrics, model)

class MetricsCallback:
    """Training callback for metrics collection."""
    
    def __init__(
        self,
        metrics: Union[TrainingMetrics, DistributedMetrics]
    ):
        self.metrics = metrics
    
    async def on_batch_end(
        self,
        step: int,
        metrics: Dict[str, float],
        model: torch.nn.Module
    ) -> None:
        """Update metrics after each batch."""
        await self.metrics.update_metrics(step, metrics, model)
    
    async def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Generate visualizations after each epoch."""
        if self.metrics.config.plot_metrics:
            self.metrics.plot_metrics(
                save_dir=Path(f"figures/epoch_{epoch}")
            )
        
        self.metrics.save_metrics(
            save_path=Path(f"metrics/epoch_{epoch}.json")
        ) 