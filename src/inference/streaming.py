"""
Streaming inference implementation.
"""

from typing import AsyncIterator

from .config import InferenceConfig
from .pipeline import InferencePipeline


class StreamingInference:
    """Streaming inference implementation."""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize streaming inference.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.pipeline = InferencePipeline(config)

    async def generate_stream(self, text: str) -> AsyncIterator[str]:
        """Generate streaming response.

        Args:
            text: Input text

        Yields:
            Generated tokens
        """
        async for token in self.pipeline.generate_stream(text):
            yield token
