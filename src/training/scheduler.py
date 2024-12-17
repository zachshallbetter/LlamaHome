"""
Learning rate scheduler implementation for training pipeline.
"""

from typing import Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def create_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> _LRScheduler:
    """Create learning rate scheduler based on configuration."""
    scheduler_type = config.get("scheduler", {}).get("type", "StepLR")
    scheduler_params = config.get("scheduler", {}).get("params", {})

    if scheduler_type == "StepLR":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, **scheduler_params)
    elif scheduler_type == "ExponentialLR":
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_type == "CosineAnnealingLR":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == "ReduceLROnPlateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
