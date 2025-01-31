"""Basic inference example."""

import asyncio
from pathlib import Path

import torch

from src.inference import InferenceConfig, InferencePipeline


async def run_inference() -> None:
    """Run basic inference example."""
    config = InferenceConfig(
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    pipeline = InferencePipeline(config)

    prompt = "Once upon a time"
    response = await pipeline.generate(prompt)
    print(f"Input: {prompt}")
    print(f"Output: {response}")


if __name__ == "__main__":
    asyncio.run(run_inference())
