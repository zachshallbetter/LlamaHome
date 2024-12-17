"""
Example script demonstrating training pipeline usage.
"""

import asyncio
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training import (
    CacheConfig,
    DataConfig,
    MonitorConfig,
    OptimizationConfig,
    ProcessingConfig,
    ResourceConfig,
    TrainingConfig,
    TrainingPipeline
)

async def main():
    """Run training example."""
    # Configuration
    config = TrainingConfig(
        cache=CacheConfig(
            memory_size=1000,
            disk_size=10000,
            use_mmap=True,
            compression=True
        ),
        data=DataConfig(
            batch_size=4,
            max_length=512,
            num_workers=4,
            validation_split=0.1
        ),
        monitor=MonitorConfig(
            tensorboard=True,
            progress_bars=True,
            resource_monitoring=True
        ),
        optimization=OptimizationConfig(
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            scheduler_type="cosine"
        ),
        processing=ProcessingConfig(
            mixed_precision=True,
            gradient_checkpointing=True,
            max_batch_size=8
        ),
        resource=ResourceConfig(
            gpu_memory_fraction=0.9,
            cpu_usage_threshold=0.8,
            io_queue_size=1000
        ),
        num_epochs=3,
        save_steps=1000,
        eval_steps=100,
        logging_steps=10,
        output_dir="output/training",
        cache_dir=".cache/training",
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-1.3b",
        padding_side="left"
    )
    
    # Initialize pipeline
    print("Initializing training pipeline...")
    pipeline = TrainingPipeline(model, tokenizer, config)
    
    # Training data
    train_data = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is machine learning?"
                },
                {
                    "role": "assistant",
                    "content": "Machine learning is a branch of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "How does deep learning work?"
                },
                {
                    "role": "assistant",
                    "content": "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input."
                }
            ]
        }
    ] * 50  # Replicate for more training data
    
    # Evaluation data
    eval_data = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is reinforcement learning?"
                },
                {
                    "role": "assistant",
                    "content": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties."
                }
            ]
        }
    ] * 10  # Replicate for more evaluation data
    
    # Save data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    import json
    with open(data_dir / "train.json", "w") as f:
        json.dump(train_data, f)
    with open(data_dir / "eval.json", "w") as f:
        json.dump(eval_data, f)
    
    # Train model
    print("Starting training...")
    try:
        await pipeline.train(
            data_dir / "train.json",
            data_dir / "eval.json"
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    # Run example
    asyncio.run(main()) 