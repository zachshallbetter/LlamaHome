"""
Distributed training implementation.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .monitoring import DistributedMetrics
from .data import StreamingDataset, DataConfig
from .processing import ProcessingConfig, TensorProcessor

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


class DistributedTrainer:
    """Distributed training implementation."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[DistributedConfig] = None,
        data_config: Optional[DataConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
    ):
        self.model = model
        self.config = config or DistributedConfig()
        self.data_config = data_config or DataConfig()
        self.processing_config = processing_config or ProcessingConfig()
        self._setup_distributed()

    def _setup_distributed(self) -> None:
        """Set up distributed training environment."""
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
            )

    def _setup_model(self) -> None:
        """Set up distributed model."""
        if self.config.sync_batch_norm:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph,
        )

    async def _setup_data(
        self, dataset: StreamingDataset
    ) -> torch.utils.data.DataLoader:
        """Set up distributed data loading."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=self.data_config.shuffle,
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            sampler=sampler,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            prefetch_factor=self.data_config.prefetch_factor,
        )

    async def train(
        self,
        dataset: StreamingDataset,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Run distributed training."""
        # Set up distributed components
        self._setup_model()
        dataloader = await self._setup_data(dataset)
        processor = TensorProcessor(self.model, self.processing_config)

        # Set up metrics
        metrics = DistributedMetrics(
            model_name=self.model.__class__.__name__,
            world_size=self.config.world_size,
            rank=self.config.rank,
        )

        # Training loop
        for epoch in range(num_epochs):
            dataloader.sampler.set_epoch(epoch)
            epoch_metrics = await self._train_epoch(
                dataloader, processor, optimizer, scheduler, metrics
            )

            # Save checkpoint on main process
            if self.config.rank == 0 and checkpoint_dir:
                await self._save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch}.pt", epoch, epoch_metrics
                )

            # Synchronize processes
            dist.barrier()

        return metrics.metrics_history

    async def _train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        processor: TensorProcessor,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: DistributedMetrics,
    ) -> Dict[str, float]:
        """Train single epoch with distributed support."""
        epoch_metrics = {}
        self.model.train()

        for step, batch in enumerate(dataloader):
            # Process batch
            optimizer.zero_grad()
            batch_metrics = await processor.process_batch(batch, is_training=True)

            # Backward pass
            batch_metrics["loss"].backward()

            # Gradient synchronization
            if self.config.gradient_as_bucket_view:
                optimizer.step()
            else:
                for param in self.model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= self.config.world_size
                optimizer.step()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Update metrics
            await metrics.update_metrics(
                step + len(dataloader) * epoch, batch_metrics, self.model
            )

            # Accumulate epoch metrics
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0
                epoch_metrics[key] += value.item() if torch.is_tensor(value) else value

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)

        return epoch_metrics

    async def _save_checkpoint(
        self, path: Path, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state": self.model.module.state_dict(),
            "metrics": metrics,
            "config": {
                "distributed": self.config,
                "data": self.data_config,
                "processing": self.processing_config,
            },
        }

        torch.save(state, path)

    async def load_checkpoint(self, path: Path) -> Dict:
        """Load training checkpoint."""
        state = torch.load(path, map_location=f"cuda:{self.config.local_rank}")

        # Load model state
        self.model.module.load_state_dict(state["model_state"])

        # Synchronize model parameters
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

        return state

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()


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
