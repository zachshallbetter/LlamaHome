"""
Example script demonstrating training with custom datasets in LlamaHome.
"""

import asyncio
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training import (
    CacheConfig,
    DataConfig,
    MonitorConfig,
    OptimizationConfig,
    ProcessingConfig,
    ResourceConfig,
    TrainingConfig,
    TrainingPipeline,
)


class CustomDataset(Dataset):
    """Example custom dataset implementation."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Combine messages into a single string
        text = ""
        for msg in item["messages"]:
            if msg["role"] == "system":
                text += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                text += f"Assistant: {msg['content']}\n"

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze(),
        }


async def main():
    """Run custom dataset training example."""
    # Configuration
    config = TrainingConfig(
        cache=CacheConfig(memory_size=1000, disk_size=10000, use_mmap=True),
        data=DataConfig(
            batch_size=4, max_length=512, num_workers=4, validation_split=0.1
        ),
        monitor=MonitorConfig(
            tensorboard=True, progress_bars=True, resource_monitoring=True
        ),
        optimization=OptimizationConfig(
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            scheduler_type="cosine",
        ),
        processing=ProcessingConfig(
            mixed_precision=True, gradient_checkpointing=True, max_batch_size=8
        ),
        resource=ResourceConfig(
            gpu_memory_fraction=0.9, cpu_usage_threshold=0.8, io_queue_size=1000
        ),
        num_epochs=3,
        save_steps=1000,
        eval_steps=100,
        logging_steps=10,
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")

    # Create custom datasets
    print("Preparing custom datasets...")

    # Example custom data
    train_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "How do I read a file in Python?"},
                {
                    "role": "assistant",
                    "content": "You can use the built-in open() function:\n\nwith open('file.txt', 'r') as f:\n    content = f.read()",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Explain list comprehension"},
                {
                    "role": "assistant",
                    "content": "List comprehension is a concise way to create lists in Python:\n\nnumbers = [x*2 for x in range(5)]",
                },
            ]
        },
    ] * 50  # Replicate for more training data

    eval_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "What is a dictionary in Python?"},
                {
                    "role": "assistant",
                    "content": "A dictionary is a collection of key-value pairs:\n\nmy_dict = {'key': 'value', 'number': 42}",
                },
            ]
        }
    ] * 10  # Replicate for more evaluation data

    # Create custom datasets
    train_dataset = CustomDataset(train_data, tokenizer, config.data.max_length)
    eval_dataset = CustomDataset(eval_data, tokenizer, config.data.max_length)

    # Initialize pipeline
    print("Initializing training pipeline...")
    pipeline = TrainingPipeline(
        model, tokenizer, config, train_dataset=train_dataset, eval_dataset=eval_dataset
    )

    # Train model
    print("Starting training...")
    try:
        await pipeline.train()
        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
