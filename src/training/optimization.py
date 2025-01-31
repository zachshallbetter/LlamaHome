"""Training optimization utilities."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

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


@dataclass
class OptimizerConfig:
    """Configuration for training optimization."""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    optimizer_type: str = "adamw"


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
    """Training optimizer with enhanced features."""

    def __init__(self, config: OptimizerConfig):
        """Initialize optimizer.

        Args:
            config: Optimizer configuration
        """
        self.config = config
        self.optimizer: Optional[Optimizer] = None

    def _setup_optimization(self) -> None:
        """Set up optimizer and scheduler."""
        # Create parameter groups
        self.param_groups = self._create_param_groups()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = None  # Set later with num_training_steps

    def _create_param_groups(self) -> List[Dict]:
        """Create parameter groups for optimization."""
        no_decay = ["bias", "LayerNorm.weight"]

        return [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with configuration."""
        return torch.optim.AdamW(
            self.param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def setup_scheduler(self, num_training_steps: int) -> None:
        """Set up learning rate scheduler."""
        self.scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def step(self, loss: torch.Tensor, update_scheduler: bool = True) -> None:
        """
        Optimization step with gradient clipping.

        Args:
            loss: Loss tensor
            update_scheduler: Whether to update scheduler
        """
        # Backward pass
        loss.backward()

        # Clip gradients
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if update_scheduler and self.scheduler is not None:
            self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        return [self.config.learning_rate]

    def state_dict(self) -> Dict:
        """Get optimizer and scheduler state."""
        state = {
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer and scheduler state."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])


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
