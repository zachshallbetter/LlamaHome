"""
Distributed training implementation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.resource.config import GPUConfig
from ..core.utils.io import safe_torch_load, safe_torch_save
from .data import StreamingDataset
from .monitoring import DistributedMetrics
from .pipeline import TrainingConfig
from .processing import TensorProcessor

# Add epoch
epoch: int = 0


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    sync_batch_norm: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    num_nodes: int = 1
    node_rank: int = 0


@dataclass
class DistributedTrainerConfig:
    """Configuration for distributed training."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    training_config: TrainingConfig
    gpu_config: GPUConfig
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    sync_batch_norm: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False


class DistributedTrainer:
    """Manages distributed training across multiple processes."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize distributed trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.world_size = config["distributed"]["world_size"]
        self.rank = config["distributed"]["rank"]
        self._initialize_process_group()

    def _initialize_process_group(self):
        """Initialize the distributed process group."""
        dist.init_process_group(
            backend=self.config["distributed"]["backend"],
            init_method=self.config["distributed"]["init_method"],
            world_size=self.world_size,
            rank=self.rank,
        )

    def distribute_model(self, model: nn.Module) -> DistributedDataParallel:
        """Wrap model for distributed training.
        
        Args:
            model: Model to distribute
            
        Returns:
            Distributed model
        """
        device = self.get_device()
        model = model.to(device)
        return DistributedDataParallel(model, device_ids=[device.index])

    def create_distributed_sampler(self, dataset) -> 'DistributedSampler':
        """Create a distributed sampler for the dataset.
        
        Args:
            dataset: Dataset to sample from
            
        Returns:
            Distributed sampler
        """
        return DistributedSampler(
            dataset_size=len(dataset),
            num_replicas=self.world_size,
            rank=self.rank
        )

    def synchronize_gradients(self, model: nn.Module) -> float:
        """Synchronize gradients across processes.
        
        Args:
            model: Model to synchronize
            
        Returns:
            Global gradient norm
        """
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
        return self._compute_grad_norm(model)

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute gradient norm.
        
        Args:
            model: Model to compute norm for
            
        Returns:
            Gradient norm
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        path: Path,
    ):
        """Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            path: Path to save to
        """
        if self.rank == 0:  # Only save on main process
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
            }, path)

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load training checkpoint.
        
        Args:
            path: Path to load from
            
        Returns:
            Checkpoint state
        """
        return torch.load(path, map_location=self.get_device())

    def get_device(self) -> torch.device:
        """Get device for current process.
        
        Returns:
            Torch device
        """
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
        return torch.device('cpu')

    def validate_world_size(self, size: int):
        """Validate world size parameter.
        
        Args:
            size: World size to validate
            
        Raises:
            ValueError: If size is invalid
        """
        if size <= 0:
            raise ValueError(f"World size must be positive, got {size}")

    def validate_rank(self, rank: int):
        """Validate rank parameter.
        
        Args:
            rank: Rank to validate
            
        Raises:
            ValueError: If rank is invalid
        """
        if rank < 0:
            raise ValueError(f"Rank must be non-negative, got {rank}")

    def synchronize(self):
        """Synchronize all processes."""
        dist.barrier()


class DistributedSampler:
    """Sampler for distributed training."""

    def __init__(self, dataset_size: int, num_replicas: int, rank: int):
        """Initialize distributed sampler.
        
        Args:
            dataset_size: Total dataset size
            num_replicas: Number of processes
            rank: Current process rank
        """
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = self._get_num_samples()
        self.total_size = self._get_total_size()

    def _get_num_samples(self) -> int:
        """Calculate number of samples for this process.
        
        Returns:
            Number of samples
        """
        return (self.dataset_size - self.rank - 1) // self.num_replicas + 1

    def _get_total_size(self) -> int:
        """Calculate total size across all processes.
        
        Returns:
            Total size
        """
        return self.num_samples * self.num_replicas

    def __iter__(self):
        """Get iterator over indices.
        
        Returns:
            Iterator over indices
        """
        # Deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.dataset_size, generator=g).tolist()

        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        """Get length of sampler.
        
        Returns:
            Number of samples
        """
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set the epoch number.
        
        Args:
            epoch: Epoch number
        """
        self.epoch = epoch


class GradientSynchronizer:
    """Handles gradient synchronization across processes."""

    def __init__(self, world_size: int):
        """Initialize gradient synchronizer.
        
        Args:
            world_size: Number of processes
        """
        self.world_size = world_size

    def synchronize(
        self,
        gradients: List[torch.Tensor],
        reduction: str = "mean"
    ) -> List[torch.Tensor]:
        """Synchronize gradients across processes.
        
        Args:
            gradients: List of gradient tensors
            reduction: Reduction operation ("sum" or "mean")
            
        Returns:
            Synchronized gradients
        """
        for grad in gradients:
            if reduction == "sum":
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            elif reduction == "mean":
                dist.all_reduce(grad, op=dist.ReduceOp.AVG)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

        return gradients


def launch_distributed(
    fn, world_size: int, num_nodes: int = 1, node_rank: int = 0, *args, **kwargs
) -> None:
    """Launch distributed training processes."""
    mp.spawn(
        fn,
        args=(world_size, num_nodes, node_rank, *args),
        nprocs=world_size // num_nodes,
        join=True,
    )


class DistributedError(Exception):
    """Distributed training error."""

    pass
