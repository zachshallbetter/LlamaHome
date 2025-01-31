"""
Manager class for inference.
"""

from typing import Any, List, Optional

from .config import InferenceConfig
from .pipeline import InferencePipeline
from .streaming import StreamingPipeline


class InferenceManager:
    """Manager class for handling inference operations."""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize inference manager.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.pipeline = (
            StreamingPipeline(config)
            if config.stream_output
            else InferencePipeline(config)
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        return self.pipeline.generate(prompt, **kwargs)

    def generate_batch(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated text responses
        """
        return self.pipeline.generate_batch(prompts, **kwargs)

    async def generate_stream(
        self, prompt: str, chunk_size: Optional[int] = None, **kwargs: Any
    ) -> Any:
        """Generate streaming response for a prompt.

        Args:
            prompt: Input prompt
            chunk_size: Size of response chunks
            **kwargs: Additional generation parameters

        Returns:
            AsyncIterator of generated text chunks

        Raises:
            RuntimeError: If streaming is not enabled
        """
        if not isinstance(self.pipeline, StreamingPipeline):
            raise RuntimeError("Streaming not enabled in configuration")

        chunk_size = chunk_size or 8
        return self.pipeline.generate_stream(prompt, chunk_size=chunk_size, **kwargs)
