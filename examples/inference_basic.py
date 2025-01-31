"""
Example script demonstrating basic inference with LlamaHome.
"""

import asyncio
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference import (
    CacheConfig,
    InferenceConfig,
    ProcessingConfig,
    ResourceConfig,
    InferencePipeline,
)


async def main():
    """Run inference example."""
    # Configuration
    config = InferenceConfig(
        cache=CacheConfig(memory_size=1000, disk_size=10000, use_mmap=True),
        processing=ProcessingConfig(
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=1,
            batch_size=1,
        ),
        resource=ResourceConfig(
            gpu_memory_fraction=0.9, cpu_usage_threshold=0.8, io_queue_size=1000
        ),
    )

    # Initialize pipeline
    print("Initializing inference pipeline...")
    pipeline = InferencePipeline(model_name="facebook/opt-1.3b", config=config)

    # Example prompts
    prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks.",
        "How does deep learning differ from traditional machine learning?",
    ]

    # Run inference
    print("Running inference...")
    try:
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            response = await pipeline.generate(prompt)
            print(f"Response: {response}")

    except Exception as e:
        print(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
