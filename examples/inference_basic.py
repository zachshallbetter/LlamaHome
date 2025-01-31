"""Basic inference example."""

from src.inference import InferenceManager
from src.inference.config import InferenceConfig

# Create inference configuration
# ruff: noqa: E501
config = InferenceConfig(
    model_name="llama3.3-7b",
    batch_size=4,
    max_length=1024,
    temperature=0.8
)

# Initialize inference manager
manager = InferenceManager(config)

# Run inference
text = "Summarize this article:"
result = manager.generate(text)
print(result)
