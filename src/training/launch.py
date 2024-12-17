"""
Distributed training launcher.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import toml
from torch.distributed.elastic.multiprocessing.errors import record

from .distributed import DistributedTrainer, DistributedConfig
from .data import DataConfig, StreamingDataset
from .processing import ProcessingConfig
from .monitoring import MetricsConfig
from .model import create_model  # Assuming create_model is defined in model module
from .optimization import create_optimizer  # Assuming create_optimizer is defined in optimizer module
from .scheduler import create_scheduler  # Assuming create_scheduler is defined in scheduler module
from .utils import save_results  # Assuming save_results is defined in utils module

def load_config(config_path: Path) -> Dict:
    """Load configuration from file."""
    with open(config_path) as f:
        return toml.load(f)

def setup_environment(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str
) -> None:
    """Set up distributed environment."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())

@record
def train_worker(
    rank: int,
    world_size: int,
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    num_epochs: int,
    master_addr: str = "localhost",
    master_port: str = "29500",
    node_rank: int = 0,
    num_nodes: int = 1
) -> None:
    """Worker process for distributed training."""
    # Load configuration
    config = load_config(config_path)
    
    # Set up environment
    setup_environment(
        rank + node_rank * (world_size // num_nodes),
        world_size,
        master_addr,
        master_port
    )
    
    # Create configurations
    distributed_config = DistributedConfig(
        world_size=world_size,
        rank=rank + node_rank * (world_size // num_nodes),
        local_rank=rank,
        master_addr=master_addr,
        master_port=master_port,
        num_nodes=num_nodes,
        node_rank=node_rank,
        **config["distributed"]
    )
    
    data_config = DataConfig(**config["resources"])
    processing_config = ProcessingConfig(**config["optimization"])
    
    # Initialize model and move to device
    model = create_model()  # Implement based on your model
    model.to(f"cuda:{rank}")
    
    # Create trainer
    trainer = DistributedTrainer(
        model,
        distributed_config,
        data_config,
        processing_config
    )
    
    try:
        # Load dataset
        dataset = StreamingDataset(
            data_path,
            buffer_size=config["resources"]["prefetch_factor"],
            memory_limit=config["resources"]["max_memory"]
        )
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)  # Implement based on your needs
        scheduler = create_scheduler(optimizer, config)  # Implement based on your needs
        
        # Train model
        metrics = trainer.train(
            dataset,
            num_epochs,
            optimizer,
            scheduler,
            output_dir / "checkpoints"
        )
        
        # Save final results on main process
        if rank == 0:
            save_results(output_dir, metrics)
            
    except Exception as e:
        print(f"Error in worker {rank}: {e}")
        raise
    finally:
        trainer.cleanup()

def launch_distributed(
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    num_epochs: int,
    world_size: Optional[int] = None,
    num_nodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: str = "29500"
) -> None:
    """Launch distributed training."""
    # Determine world size if not specified
    if world_size is None:
        world_size = torch.cuda.device_count() * num_nodes
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Launch processes
    if num_nodes > 1:
        # Multi-node training
        local_world_size = world_size // num_nodes
        torch.multiprocessing.spawn(
            train_worker,
            args=(
                local_world_size,
                config_path,
                data_path,
                output_dir,
                num_epochs,
                master_addr,
                master_port,
                node_rank,
                num_nodes
            ),
            nprocs=local_world_size
        )
    else:
        # Single-node training
        torch.multiprocessing.spawn(
            train_worker,
            args=(
                world_size,
                config_path,
                data_path,
                output_dir,
                num_epochs,
                master_addr,
                master_port,
                0,
                1
            ),
            nprocs=world_size
        )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed training launcher")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--data", type=Path, required=True, help="Path to data")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--world-size", type=int, help="Total number of processes")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="Current node rank")
    parser.add_argument("--master-addr", default="localhost", help="Master node address")
    parser.add_argument("--master-port", default="29500", help="Master node port")
    
    args = parser.parse_args()
    
    launch_distributed(
        args.config,
        args.data,
        args.output,
        args.epochs,
        args.world_size,
        args.num_nodes,
        args.node_rank,
        args.master_addr,
        args.master_port
    )

if __name__ == "__main__":
    main() 