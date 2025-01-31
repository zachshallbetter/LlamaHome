"""Example of training with custom data in LlamaHome."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingPipeline,
)


@dataclass
class CustomDataConfig:
    """Custom data configuration."""

    input_files: List[str]
    max_length: int = 512
    batch_size: int = 32
    num_workers: int = 4


def prepare_dataset(config: CustomDataConfig) -> Dict:
    """Prepare custom dataset for training."""
    return {"data": []}


async def train_custom() -> None:
    """Run custom data training example."""
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("llama-3.3-7b")
    tokenizer = AutoTokenizer.from_pretrained("llama-3.3-7b")

    # Configure training
    training_config = TrainingConfig(
        batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_epochs=3,
        optimizer=OptimizerConfig(name="adamw", weight_decay=0.01),
        scheduler=SchedulerConfig(name="cosine", num_warmup_steps=100),
    )

    # Initialize pipeline
    pipeline = TrainingPipeline(
        model=model, tokenizer=tokenizer, config=training_config
    )

    try:
        # Start training
        await pipeline.train(
            train_data="data/custom/train.txt", eval_data="data/custom/eval.txt"
        )

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(train_custom())
