"""Streaming inference example."""

import asyncio

from src.core.config import ConfigManager
from src.inference import InferenceConfig, StreamingInference


async def process_stream() -> None:
    """Process streaming inference."""
    # Get configuration
    ConfigManager()
    inference_config = InferenceConfig()
    inference_config.stream_output = True

    # Initialize streaming inference
    inference = StreamingInference(inference_config)

    # Run streaming inference
    text = "Write a story about:"
    async for token in inference.generate_stream(text):
        print(token, end="", flush=True)


def main() -> None:
    """Run streaming example."""
    asyncio.run(process_stream())


if __name__ == "__main__":
    main()
