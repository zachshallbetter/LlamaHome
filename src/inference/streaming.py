"""
Streaming inference implementation.
"""

import asyncio
from typing import Any, AsyncIterator, List

from .config import InferenceConfig
from .pipeline import InferencePipeline


class StreamingPipeline(InferencePipeline):
    """Pipeline for streaming inference responses."""

    async def generate_stream(
        self, prompt: str, chunk_size: int = 8, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Generate streaming response for a given prompt.

        Args:
            prompt: Input prompt
            chunk_size: Size of response chunks
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks

        Raises:
            RuntimeError: If streaming generation fails
        """
        try:
            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.processing.max_length,
            ).to(self.config.device)

            # Generate with resource management
            async with self.resource_manager.optimize():
                # Initialize generation
                generated_tokens: List[int] = []

                # Stream generation
                for _ in range(0, self.config.processing.max_length, chunk_size):
                    # Generate next chunk
                    outputs = await asyncio.to_thread(
                        self.model.generate,
                        **inputs,
                        max_new_tokens=chunk_size,
                        temperature=self.config.processing.temperature,
                        top_p=self.config.processing.top_p,
                        top_k=self.config.processing.top_k,
                        num_beams=1,  # Streaming works better with greedy decoding
                        do_sample=self.config.processing.do_sample,
                        repetition_penalty=self.config.processing.repetition_penalty,
                        length_penalty=self.config.processing.length_penalty,
                        early_stopping=self.config.processing.early_stopping,
                        **kwargs,
                    )

                    # Get new tokens
                    new_tokens = outputs[0][len(generated_tokens) :]
                    if len(new_tokens) == 0:
                        break

                    generated_tokens.extend(new_tokens)

                    # Decode and yield new chunk
                    chunk_text = self.tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                    if chunk_text:
                        yield chunk_text

                    # Check for end of generation
                    if self.tokenizer.eos_token_id in new_tokens:
                        break

                    # Update input_ids for next iteration
                    inputs["input_ids"] = outputs

        except Exception as e:
            raise RuntimeError(f"Streaming generation failed: {str(e)}") from e
