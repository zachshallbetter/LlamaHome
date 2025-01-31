"""Checkpoint management functionality."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    
    save_dir: str = "checkpoints"
    save_steps: int = 1000
    save_epochs: int = 1
    keep_last_n: int = 5
    save_best: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_metrics: bool = True
    metric_for_best: str = "loss"
    greater_is_better: bool = False
    save_format: str = "pytorch"  # or "safetensors"


class CheckpointManager:
    """Manages model checkpoints and training state."""

    def __init__(self, config: Dict[str, Any], model_name: str):
        """Initialize checkpoint manager.
        
        Args:
            config: Configuration dictionary
            model_name: Name of the model
        """
        self.config = CheckpointConfig(**config.get("checkpoint", {}))
        self.model_name = model_name
        self.save_dir = Path(self.config.save_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = float('inf')
        if self.config.greater_is_better:
            self.best_metric = float('-inf')
        
        self.checkpoints: List[Dict[str, Any]] = []
        self._load_checkpoint_history()

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            metrics: Optional metrics to save
            step: Optional step number
            epoch: Optional epoch number
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}"
        if step is not None:
            checkpoint_name += f"_step{step}"
        if epoch is not None:
            checkpoint_name += f"_epoch{epoch}"
        
        checkpoint_path = self.save_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)

        # Save model
        if self.config.save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                state_dict = model.state_dict()
                save_file(state_dict, checkpoint_path / "model.safetensors")
            except ImportError:
                logger.warning("safetensors not available, falling back to PyTorch format")
                torch.save(model.state_dict(), checkpoint_path / "model.pt")
        else:
            torch.save(model.state_dict(), checkpoint_path / "model.pt")

        # Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }

        if self.config.save_optimizer and optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
        
        if self.config.save_scheduler and scheduler is not None:
            training_state["scheduler"] = scheduler.state_dict()

        torch.save(training_state, checkpoint_path / "training_state.pt")

        # Update checkpoint history
        checkpoint_info = {
            "path": str(checkpoint_path),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "is_best": is_best,
            "timestamp": timestamp,
        }
        self.checkpoints.append(checkpoint_info)
        self._save_checkpoint_history()

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Optional specific checkpoint to load
            model: Optional model to load into
            optimizer: Optional optimizer to load into
            scheduler: Optional scheduler to load into
            map_location: Optional device mapping
            
        Returns:
            Loaded training state
        """
        if checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoints:
                raise ValueError("No checkpoints found")
            checkpoint_path = Path(self.checkpoints[-1]["path"])

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        # Load model
        if model is not None:
            if (checkpoint_path / "model.safetensors").exists():
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(checkpoint_path / "model.safetensors")
                    model.load_state_dict(state_dict)
                except ImportError:
                    logger.warning("safetensors not available, falling back to PyTorch format")
                    state_dict = torch.load(checkpoint_path / "model.pt", map_location=map_location)
                    model.load_state_dict(state_dict)
            else:
                state_dict = torch.load(checkpoint_path / "model.pt", map_location=map_location)
                model.load_state_dict(state_dict)

        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt", map_location=map_location)

        if optimizer is not None and "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
        
        if scheduler is not None and "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])

        return training_state

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get information about best checkpoint.
        
        Returns:
            Best checkpoint info or None
        """
        best_checkpoints = [c for c in self.checkpoints if c.get("is_best", False)]
        if not best_checkpoints:
            return None
        return best_checkpoints[-1]

    def update_best(self, metrics: Dict[str, float], checkpoint_path: Path) -> bool:
        """Update best checkpoint if metrics improved.
        
        Args:
            metrics: Current metrics
            checkpoint_path: Path to current checkpoint
            
        Returns:
            Whether this is the new best checkpoint
        """
        if not self.config.save_best or self.config.metric_for_best not in metrics:
            return False

        current_metric = metrics[self.config.metric_for_best]
        is_better = False

        if self.config.greater_is_better:
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric

        if is_better:
            self.best_metric = current_metric
            # Update checkpoint history
            for checkpoint in self.checkpoints:
                checkpoint["is_best"] = False
            self.checkpoints[-1]["is_best"] = True
            self._save_checkpoint_history()
            
            # Create symlink to best checkpoint
            best_path = self.save_dir / "best_checkpoint"
            if best_path.exists():
                best_path.unlink()
            best_path.symlink_to(checkpoint_path, target_is_directory=True)
            
            return True

        return False

    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from disk."""
        history_path = self.save_dir / "checkpoint_history.json"
        if history_path.exists():
            try:
                with open(history_path) as f:
                    self.checkpoints = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoint history: {e}")
                self.checkpoints = []

    def _save_checkpoint_history(self) -> None:
        """Save checkpoint history to disk."""
        history_path = self.save_dir / "checkpoint_history.json"
        try:
            with open(history_path, "w") as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint history: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save space."""
        if not self.config.keep_last_n or len(self.checkpoints) <= self.config.keep_last_n:
            return

        # Keep best checkpoint if it exists
        checkpoints_to_remove = []
        best_checkpoint = self.get_best_checkpoint()
        
        for checkpoint in self.checkpoints[:-self.config.keep_last_n]:
            if best_checkpoint and checkpoint["path"] == best_checkpoint["path"]:
                continue
            checkpoints_to_remove.append(checkpoint)

        # Remove old checkpoints
        for checkpoint in checkpoints_to_remove:
            try:
                path = Path(checkpoint["path"])
                if path.exists():
                    import shutil
                    shutil.rmtree(path)
                self.checkpoints.remove(checkpoint)
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {path}: {e}")

        self._save_checkpoint_history()
