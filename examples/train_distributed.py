"""
Example script demonstrating distributed training across multiple GPUs.
"""

import asyncio
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training import (
    CacheConfig,
    DataConfig,
    MonitorConfig,
    OptimizationConfig,
    ProcessingConfig,
    ResourceConfig,
    TrainingConfig,
    DistributedTrainingPipeline,
)


async def main():
    """Run distributed training example."""
    # Initialize distributed environment
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Distributed training requires at least 2 GPUs")

    # Configuration
    config = TrainingConfig(
        cache=CacheConfig(
            memory_size=1000 // world_size,  # Split memory across GPUs
            disk_size=10000,
            use_mmap=True,
            compression=True,
        ),
        data=DataConfig(
            batch_size=4 * world_size,  # Scale batch size with GPUs
            max_length=512,
            num_workers=4,
            validation_split=0.1,
        ),
        monitor=MonitorConfig(
            tensorboard=True, progress_bars=True, resource_monitoring=True
        ),
        optimization=OptimizationConfig(
            learning_rate=5e-5 * world_size,  # Scale learning rate
            weight_decay=0.01,
            warmup_steps=100,
            scheduler_type="cosine",
        ),
        processing=ProcessingConfig(
            mixed_precision=True,
            gradient_checkpointing=True,
            max_batch_size=8 * world_size,  # Scale max batch size
        ),
        resource=ResourceConfig(
            gpu_memory_fraction=0.9, cpu_usage_threshold=0.8, io_queue_size=1000
        ),
        num_epochs=3,
        save_steps=1000,
        eval_steps=100,
        logging_steps=10,
        output_dir="output/distributed_training",
        cache_dir=".cache/distributed_training",
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )

    # Initialize distributed process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=world_size,
        rank=0,  # Main process rank
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")

    # Initialize distributed pipeline
    print("Initializing distributed training pipeline...")
    pipeline = DistributedTrainingPipeline(
        model, tokenizer, config, world_size=world_size
    )

    # Use same training data as train_model.py
    data_dir = Path("data")

    # Train model
    print("Starting distributed training...")
    try:
        await pipeline.train(data_dir / "train.json", data_dir / "eval.json")
        print("Distributed training completed successfully!")

    except Exception as e:
        print(f"Distributed training failed: {e}")
        raise

    finally:
        # Cleanup
        dist.destroy_process_group()


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
