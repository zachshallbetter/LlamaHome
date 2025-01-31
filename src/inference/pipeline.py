"""
Inference pipeline implementation.
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import InferenceConfig
from ..core import ResourceManager, MemoryTracker


class InferencePipeline:
    """Main inference pipeline."""

    def __init__(self, model_name: str, config: InferenceConfig, **kwargs):
        """Initialize inference pipeline."""
        self.model_name = model_name
        self.config = config
        self.resource_manager = ResourceManager(config.resource)
        self.memory_tracker = MemoryTracker()

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model and tokenizer with proper configuration."""
        # Set device
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=self.config.trust_remote_code,
            use_auth_token=self.config.use_auth_token,
            revision=self.config.model_revision,
            **({"load_in_8bit": True} if self.config.quantization == "int8" else {}),
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=self.config.trust_remote_code,
            use_auth_token=self.config.use_auth_token,
            revision=self.config.model_revision,
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response for a given prompt."""
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
                # Generate response
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

            # Decode response
            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            return response

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}") from e

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
            ).to(self.config.device)

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
