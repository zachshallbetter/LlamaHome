"""
Distributed training launcher.
"""

import asyncio
import logging
import os
import sys

from pathlib import Path
from typing import Any, Dict, Optional

import toml
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ..core.config import TrainingConfig
from ..core.config.base import BaseConfig
from .data import DataConfig, StreamingDataset
from .distributed import DistributedConfig, DistributedTrainer
from .model import create_model, ModelManager
from .optimization import create_optimizer, create_scheduler, OptimizerManager
from .pipeline import TrainingPipeline
from .processing import ProcessingConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration dictionary
    """
    try:
        return toml.load(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def setup_environment(
    rank: int, world_size: int, master_addr: str, master_port: str
) -> None:
    """Set up distributed environment."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())


@record
async def train_worker(
    rank: int,
    world_size: int,
    config: TrainingConfig,
    data_path: Path,
    output_dir: Optional[Path] = None,
) -> None:
    """Training worker process.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration
        data_path: Path to training data
        output_dir: Optional output directory
    """
    try:
        # Set up distributed environment
        setup_environment(
            rank=rank,
            world_size=world_size,
            master_addr=config.distributed.master_addr,
            master_port=config.distributed.master_port,
        )

        # Initialize process group
        torch.distributed.init_process_group(
            backend=config.distributed.backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        # Initialize model manager
        model_manager = ModelManager(config.model)
        model = await model_manager.initialize_model()
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Initialize distributed trainer
        trainer = DistributedTrainer(
            config={
                "distributed": {
                    "world_size": world_size,
                    "rank": rank,
                    "backend": config.distributed.backend,
                    "init_method": "env://",
                }
            }
        )
        model = trainer.distribute_model(model)

        # Create dataset
        dataset = StreamingDataset(data_path)

        # Initialize optimizer manager
        optimizer_manager = OptimizerManager(config.optimization)
        optimizer = await optimizer_manager.create_optimizer(model)
        scheduler = await optimizer_manager.create_scheduler()

        # Create training pipeline
        pipeline = TrainingPipeline(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            output_dir=output_dir,
        )

        # Train model
        await pipeline.train(dataset)

    except Exception as e:
        logger.error(f"Training worker failed: {e}")
        raise


async def launch_training(
    config_path: Path,
    data_path: Path,
    output_dir: Optional[Path] = None,
    distributed: bool = False,
) -> None:
    """Launch training process.

    Args:
        config_path: Path to configuration file
        data_path: Path to training data
        output_dir: Optional output directory
        distributed: Whether to use distributed training
    """
    # Load configuration
    config = TrainingConfig.parse_obj(load_config(config_path))

    if distributed:
        world_size = torch.cuda.device_count()
        if world_size > 1:
            await asyncio.gather(*[
                train_worker(
                    rank=i,
                    world_size=world_size,
                    config=config,
                    data_path=data_path,
                    output_dir=output_dir,
                )
                for i in range(world_size)
            ])
        else:
            logger.warning("Distributed training requested but only one GPU available")
            await train_worker(0, 1, config, data_path, output_dir)
    else:
        # Initialize model manager
        model_manager = ModelManager(config.model)
        model = await model_manager.initialize_model()

        # Initialize optimizer manager
        optimizer_manager = OptimizerManager(config.optimization)
        optimizer = await optimizer_manager.create_optimizer(model)
        scheduler = await optimizer_manager.create_scheduler()

        # Create training pipeline
        pipeline = TrainingPipeline(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            output_dir=output_dir,
        )

        # Train model
        await pipeline.train(data_path)


def main() -> None:
    """Main training launch entry point."""
    try:
        config_path = Path("config/training_config.toml")
        data_path = Path("data/training")
        output_dir = Path("outputs")

        asyncio.run(launch_training(
            config_path=config_path,
            data_path=data_path,
            output_dir=output_dir,
            distributed=torch.cuda.device_count() > 1,
        ))
    except Exception as e:
        logger.error(f"Training launch error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
