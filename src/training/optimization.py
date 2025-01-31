"""Training optimization utilities."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau

from ..core.config import ConfigManager
from ..core.schemas import TrainingSchema
from ..core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.optimization_config = self.config_manager.configs["training"][
            "optimization"
        ]

        # Load settings
        self.learning_rate = float(self.optimization_config["learning_rate"])
        self.weight_decay = float(self.optimization_config["weight_decay"])
        self.warmup_steps = int(self.optimization_config["warmup_steps"])
        self.scheduler_type = self.optimization_config["scheduler_type"]
        self.max_grad_norm = float(self.optimization_config["max_grad_norm"])
        self.mixed_precision = self.optimization_config["mixed_precision"]
        self.gradient_checkpointing = self.optimization_config["gradient_checkpointing"]

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate optimization configuration."""
        config_dict = {
            "optimization": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "scheduler_type": self.scheduler_type,
                "max_grad_norm": self.max_grad_norm,
                "mixed_precision": self.mixed_precision,
                "gradient_checkpointing": self.gradient_checkpointing,
            }
        }

        TrainingSchema(**config_dict)


class OptimizerConfig(BaseConfig):
    """Optimizer configuration."""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    scheduler_type: str = "linear"


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> LRScheduler:
    """Get learning rate scheduler.

    Args:
        name: Scheduler name
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler

    Raises:
        ValueError: If scheduler type is not supported
    """
    name = name.lower()

    if name == "linear":
        return LinearScheduler(optimizer, num_warmup_steps, num_training_steps)
    elif name == "cosine":
        return CosineScheduler(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    elif name == "constant":
        return ConstantScheduler(optimizer, num_warmup_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {name}")


class Optimizer:
    """Optimizer wrapper with scheduling and gradient handling."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: OptimizerConfig,
    ):
        self.model = model
        self.config = config
        self._setup_optimizer()
        self._setup_scheduler()

    def _setup_optimizer(self) -> None:
        """Set up optimizer."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )

    def _setup_scheduler(self) -> None:
        """Set up learning rate scheduler."""
        if self.config.scheduler_type == "linear":
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.warmup_steps,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.warmup_steps,
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=10,
            )
        else:
            self.scheduler = None

    def step(self) -> None:
        """Perform optimization step."""
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient clipping."""
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler and state_dict["scheduler"]:
            self.scheduler.load_state_dict(state_dict["scheduler"])


async def create_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> Optimizer:
    """Create optimizer instance.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        Configured optimizer instance
    """
    return Optimizer(model, config)


class CosineScheduler(LRScheduler):
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for current step."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]

        # Cosine decay
        progress = (self.last_epoch - self.num_warmup_steps) / (
            self.num_training_steps - self.num_warmup_steps
        )

        return [
            base_lr * 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
            for base_lr in self.base_lrs
        ]


class LinearScheduler(LRScheduler):
    """Linear learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for current step."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]

        # Linear decay
        progress = (self.last_epoch - self.num_warmup_steps) / (
            self.num_training_steps - self.num_warmup_steps
        )

        return [base_lr * (1.0 - progress) for base_lr in self.base_lrs]


class ConstantScheduler(LRScheduler):
    """Constant learning rate scheduler with warmup."""

    def __init__(
        self, optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for current step."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]

        # Constant learning rate
        return self.base_lrs


class OptimizationError(Exception):
    """Optimization error."""

    pass
