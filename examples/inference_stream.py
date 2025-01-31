"""
Example script demonstrating streaming inference with LlamaHome.
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
    StreamingPipeline,
)
from src.interfaces import CLIInterface


async def main():
    """Run streaming inference example."""
    # Configuration
    config = InferenceConfig(
        cache=CacheConfig(memory_size=1000, disk_size=10000, use_mmap=True),
        processing=ProcessingConfig(
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=1,  # Streaming works better with greedy decoding
            batch_size=1,
            chunk_size=8,  # Token chunks for streaming
        ),
        resource=ResourceConfig(
            gpu_memory_fraction=0.9, cpu_usage_threshold=0.8, io_queue_size=1000
        ),
    )

    # Initialize pipeline
    print("Initializing streaming pipeline...")
    pipeline = StreamingPipeline(model_name="facebook/opt-1.3b", config=config)

    # Example prompts
    prompts = [
        "Write a story about a space explorer discovering a new planet. Start with:",
        "Explain how neural networks work, step by step:",
        "Write a recipe for chocolate chip cookies. Include ingredients and steps:",
    ]

    # Run streaming inference
    print("\nRunning streaming inference...")
    try:
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("Response: ", end="", flush=True)

            # Stream response
            async for chunk in pipeline.generate_stream(prompt):
                print(chunk, end="", flush=True)
            print("\n")

            # Add a delay between prompts
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Streaming inference failed: {e}")
        raise

    # Demonstrate CLI interface integration
    print("\nDemonstrating CLI interface integration...")
    cli = CLIInterface()

    try:
        # Stream with progress tracking
        prompt = "Explain quantum computing in simple terms:"
        print(f"\nPrompt: {prompt}")

        async with cli.progress_bar() as progress:
            async for chunk in pipeline.generate_stream(prompt):
                progress.update(len(chunk))
                print(chunk, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"CLI streaming failed: {e}")
        raise


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
