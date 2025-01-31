"""Training optimization utilities."""

import math
from typing import Any, Dict, List, Optional, Union, cast

import torch
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    ReduceLROnPlateau,
    _LRScheduler,
)
from transformers import get_scheduler

from ..core.config.base import BaseConfig
from ..core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


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
    total_steps: int = 1000
    num_cycles: float = 0.5
    power: float = 1.0


class OptimizationManager:
    """Manages optimization components."""

    def __init__(
        self,
        model: Module,
        config: OptimizerConfig,
    ) -> None:
        """Initialize optimization manager.

        Args:
            model: Model to optimize
            config: Optimizer configuration
        """
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer instance.

        Returns:
            Configured optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
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

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _create_scheduler(self) -> _LRScheduler:
        """Create learning rate scheduler.

        Returns:
            Configured scheduler
        """
        if self.config.scheduler_type == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.total_steps,
            )
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_steps,
                eta_min=0.0,
            )
        elif self.config.scheduler_type == "plateau":
            return cast(
                _LRScheduler,
                ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=0.1,
                    patience=10,
                    verbose=True,
                ),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def step(self, loss: Optional[float] = None) -> None:
        """Perform optimization step.

        Args:
            loss: Optional loss value for plateau scheduler
        """
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if loss is not None:
                self.scheduler.step(loss)
        else:
            self.scheduler.step()

    def get_lr(self) -> List[float]:
        """Get current learning rates.

        Returns:
            List of learning rates
        """
        if isinstance(self.scheduler, ReduceLROnPlateau):
            return [group["lr"] for group in self.optimizer.param_groups]
        return self.scheduler.get_last_lr()

    def state_dict(self) -> Dict[str, Any]:
        """Get optimization state.

        Returns:
            State dictionary
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimization state.

        Args:
            state_dict: State dictionary to load
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


async def create_optimizer(
    model: Module,
    config: OptimizerConfig,
) -> OptimizationManager:
    """Create optimization manager.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        Configured optimization manager
    """
    return OptimizationManager(model, config)


class CosineScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            num_cycles: Number of cycles for cosine decay
            last_epoch: The index of last epoch
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for current step.

        Returns:
            List of learning rates
        """
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
            base_lr
            * 0.5
            * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
            for base_lr in self.base_lrs
        ]


class LinearScheduler(_LRScheduler):
    """Linear learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            last_epoch: The index of last epoch
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for current step.

        Returns:
            List of learning rates
        """
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


class OptimizationError(Exception):
    """Optimization error."""

    pass


async def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> Optimizer:
    """Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
        
    Raises:
        ValueError: If optimizer configuration is invalid
    """
    try:
        # Get optimizer parameters
        optimizer_name = config.get("name", "adamw").lower()
        lr = config.get("learning_rate", 1e-4)
        weight_decay = config.get("weight_decay", 0.01)
        beta1 = config.get("beta1", 0.9)
        beta2 = config.get("beta2", 0.999)
        eps = config.get("eps", 1e-8)

        # Create parameter groups
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Create optimizer
        if optimizer_name == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(beta1, beta2),
                eps=eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return optimizer

    except Exception as e:
        logger.error(f"Failed to create optimizer: {e}")
        raise ValueError(f"Optimizer creation failed: {e}")


async def create_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any]
) -> Optional[LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Configured scheduler or None if not specified
        
    Raises:
        ValueError: If scheduler configuration is invalid
    """
    try:
        scheduler_config = config.get("scheduler")
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get("name", "linear")
        num_training_steps = scheduler_config.get("num_training_steps", 1000)
        num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)

        scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return scheduler

    except Exception as e:
        logger.error(f"Failed to create scheduler: {e}")
        raise ValueError(f"Scheduler creation failed: {e}")


class OptimizerManager:
    """Manages optimization components."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize optimizer manager.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.optimizer = None
        self.scheduler = None

    async def create_optimizer(self, model: torch.nn.Module) -> Optimizer:
        """Create optimizer for model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Created optimizer
        """
        self.optimizer = await create_optimizer(model, self.config)
        return self.optimizer

    async def create_scheduler(self) -> Optional[LRScheduler]:
        """Create scheduler for optimizer.
        
        Returns:
            Created scheduler or None
            
        Raises:
            ValueError: If optimizer not initialized
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be created before scheduler")

        self.scheduler = await create_scheduler(self.optimizer, self.config)
        return self.scheduler

    def save_state(self, path: str) -> None:
        """Save optimizer and scheduler state.
        
        Args:
            path: Path to save to
        """
        state = {
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load optimizer and scheduler state.
        
        Args:
            path: Path to load from
        """
        state = torch.load(path)
        
        if self.optimizer and state["optimizer"]:
            self.optimizer.load_state_dict(state["optimizer"])
            
        if self.scheduler and state["scheduler"]:
            self.scheduler.load_state_dict(state["scheduler"])


class OptimizationError(Exception):
    """Optimization-related error."""
    pass
