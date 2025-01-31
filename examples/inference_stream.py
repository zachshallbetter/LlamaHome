"""Streaming inference example."""

import asyncio
from pathlib import Path

import torch

from src.inference import InferenceConfig
from src.inference.streaming import StreamingPipeline


async def run_streaming() -> None:
    """Run streaming inference example."""
    config = InferenceConfig(
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    pipeline = StreamingPipeline(config)

    prompt = "Tell me a story about"
    print(f"Input: {prompt}")
    print("Output:", end=" ", flush=True)

    async for chunk in pipeline.generate_stream(prompt):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(run_streaming())
