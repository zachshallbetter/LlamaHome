"""Example of distributed training with LlamaHome."""

import asyncio

import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.resource.config import GPUConfig
from src.training import (
    DistributedTrainer,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


async def train_distributed() -> None:
    """Run distributed training example."""
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("llama-3.3-7b")
    tokenizer = AutoTokenizer.from_pretrained("llama-3.3-7b")

    # Configure training
    training_config = TrainingConfig(
        batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_epochs=3,
        optimizer=OptimizerConfig(
            name="adamw", weight_decay=0.01, beta1=0.9, beta2=0.999
        ),
        scheduler=SchedulerConfig(name="cosine", num_warmup_steps=100),
    )

    # Configure GPU resources
    gpu_config = GPUConfig(memory_fraction=0.9, allow_growth=True)

    # Initialize distributed trainer
    trainer = DistributedTrainer(
        model=model,
        tokenizer=tokenizer,
        training_config=training_config,
        gpu_config=gpu_config,
    )

    try:
        # Start training
        await trainer.train(
            train_dataset="path/to/train/data", eval_dataset="path/to/eval/data"
        )

    except Exception as e:
        print(f"Training failed: {e}")
        raise

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(train_distributed())
