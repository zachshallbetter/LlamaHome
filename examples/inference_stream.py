"""Streaming inference example."""

from src.inference import InferenceConfig, StreamingInference
from src.core.resource import GPUConfig
from src.core.config import ModelConfig

# Load model configuration
model_config = ModelConfig()

# Create inference configuration
config = InferenceConfig(
    model_name=model_config.name,
    batch_size=1,
    max_length=2048,
    stream_output=True,
    gpu_config=GPUConfig()
)

# Initialize streaming inference
inference = StreamingInference(config)

# Run streaming inference
async def process_stream():
    text = "Write a story about:"
    async for token in inference.generate_stream(text):
        print(token, end="", flush=True)

# Run the async function
import asyncio
asyncio.run(process_stream())
