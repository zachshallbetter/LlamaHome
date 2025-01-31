"""
Inference pipeline implementation.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.resource import ResourceManager
from ..core.utils import MemoryTracker
from ..core.utils.io import safe_load_torch
from .config import InferenceConfig


class InferencePipeline:
    """Main inference pipeline."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: InferenceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.resource_manager = ResourceManager(config.resource)
        self.memory_tracker = MemoryTracker()
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Set up inference pipeline."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        # Enable model optimizations
        self.model.eval()
        if hasattr(self.model, "use_cache"):
            self.model.use_cache = self.config.use_cache

    async def generate(self, prompt: str, **kwargs: Any) -> Union[str, List[str]]:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text or list of texts
        """
        # Tokenize input
        inputs = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_new_tokens,
            ),
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        generation_config = self._get_generation_config(**kwargs)
        outputs = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.model.generate(
                **inputs,
                **generation_config,
            ),
        )

        # Decode outputs
        decoded = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            ),
        )

        return decoded[0] if len(decoded) == 1 else decoded

    def _get_generation_config(self, **kwargs: Any) -> Dict[str, Any]:
        """Get generation configuration.

        Args:
            **kwargs: Override parameters

        Returns:
            Generation configuration dictionary
        """
        config = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "length_penalty": self.config.length_penalty,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "num_return_sequences": self.config.num_return_sequences,
            "do_sample": self.config.do_sample,
            "early_stopping": self.config.early_stopping,
        }
        return {**config, **kwargs}

    async def save(self, path: str) -> None:
        """Save pipeline to disk.

        Args:
            path: Save path
        """
        state = {
            "model": self.model.state_dict(),
            "config": self.config.dict(),
        }
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: safe_load_torch(state, path),
        )

    @classmethod
    async def load(cls, path: str) -> "InferencePipeline":
        """Load pipeline from disk.

        Args:
            path: Load path

        Returns:
            Loaded pipeline
        """
        state = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: safe_load_torch(path),
        )
        config = InferenceConfig(**state["config"])
        model = PreTrainedModel.from_pretrained(config.model.model_name)
        model.load_state_dict(state["model"])
        tokenizer = PreTrainedTokenizer.from_pretrained(config.model.model_name)
        return cls(model, tokenizer, config)

    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        try:
            # Prepare batch inputs
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.processing.max_length,
            ).to(self.device)

            # Generate with resource management
            async with self.resource_manager.optimize():
                # Generate responses
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_length=self.config.processing.max_length,
                    temperature=self.config.processing.temperature,
                    top_p=self.config.processing.top_p,
                    top_k=self.config.processing.top_k,
                    num_beams=self.config.processing.num_beams,
                    do_sample=self.config.processing.do_sample,
                    repetition_penalty=self.config.processing.repetition_penalty,
                    length_penalty=self.config.processing.length_penalty,
                    early_stopping=self.config.processing.early_stopping,
                    **kwargs,
                )

            # Decode responses
            responses = [
                self.tokenizer.decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for output in outputs
            ]

            return responses

        except Exception as e:
            raise RuntimeError(f"Batch generation failed: {str(e)}") from e
