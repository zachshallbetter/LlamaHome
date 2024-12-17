"""Integration layer for language models.

This module provides the core model integration layer for LlamaHome, handling:
- Model loading and resource management
- Async inference and streaming
- H2O acceleration integration
- Configuration management
- Error handling and logging
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import torch
from transformers import LlamaConfig

from .h2o import H2OLlamaForCausalLM
from utils.model_manager import ModelManager
from utils.setup_model import ModelSetup
from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    response_length: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: list[str] = None
    max_tokens: int = 2048
    h2o_window_length: int = 512
    h2o_heavy_hitter_tokens: int = 128


class ModelHandler:
    """Handles integration with language models."""

    def __init__(
        self, model_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None,
        model_name: str = "default"
    ) -> None:
        """Initialize the model handler.

        Args:
            model_path: Path to the model files
            config: Optional configuration dictionary
            model_name: Name of the model to load
        """
        self.model_manager = ModelManager()
        self.model_setup = ModelSetup()
        self.model_name = model_name

        # Use model manager's config if no path provided
        if not model_path:
            model_config = self.model_manager.config["models"][model_name]
            self.model_path = model_config.get(
                "default_path", Path(Path.home() / ".llamahome" / "models" / model_name)
            )
        else:
            self.model_path = model_path

        self.config = ModelConfig(**config) if config else ModelConfig()
        self._model: Optional[H2OLlamaForCausalLM] = None
        self._lock = asyncio.Lock()
        self._is_loaded = False

    async def load_model(self) -> None:
        """Load the model into memory with H2O acceleration."""
        async with self._lock:
            if self._is_loaded:
                return

            try:
                logger.info("Loading model from {}", self.model_path)

                # Validate model path
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Model not found at {self.model_path}")

                # Set up model using ModelSetup
                self.model_setup.setup_model(
                    model_name=self.model_name,
                    path=str(self.model_path),
                    model_size=self.model_manager.config["models"][self.model_name].get(
                        "versions", ["default"]
                    )[0],
                    variant="chat",
                )

                # Check GPU requirements
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    required_memory = self.model_manager.config["models"][self.model_name][
                        "min_gpu_memory"
                    ].get("default", 16)
                    if gpu_memory < required_memory:
                        raise RuntimeError(
                            f"Insufficient GPU memory: {gpu_memory:.1f}GB available, {required_memory}GB required"
                        )

                # Initialize H2O-accelerated model
                llama_config = LlamaConfig(
                    num_heavy_hitter_tokens=self.config.h2o_heavy_hitter_tokens,
                    num_window_length=self.config.h2o_window_length
                )
                self._model = H2OLlamaForCausalLM(llama_config)
                self._model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                self._is_loaded = True
                logger.info(LogTemplates.MODEL_LOADED.format(model_path=str(self.model_path)))
            except Exception as e:
                logger.exception(LogTemplates.MODEL_ERROR.format(error=str(e)))
                raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        async with self._lock:
            if not self._is_loaded:
                return

            try:
                # Cleanup resources
                if self._model is not None:
                    self._model.cpu()
                    del self._model
                    torch.cuda.empty_cache()
                self._model = None
                self._is_loaded = False
                logger.info(LogTemplates.MODEL_UNLOADED.format(model_path=str(self.model_path)))
            except Exception as e:
                logger.exception(LogTemplates.MODEL_ERROR.format(error=str(e)))
                raise

    async def generate_response_async(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a response asynchronously.

        Args:
            prompt: The input prompt
            config: Optional configuration overrides

        Returns:
            Generated text response

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If prompt is invalid
            TimeoutError: If generation times out
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Model is not loaded")

        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")

        try:
            # Validate prompt length
            if len(prompt) > self.config.max_tokens:
                raise ValueError("Prompt exceeds maximum length")

            # Generate response using H2O-accelerated model
            inputs = self._model.prepare_inputs_for_generation(
                input_ids=torch.tensor([self.model_setup.tokenizer.encode(prompt)]),
                attention_mask=None,
                use_cache=True
            )
            
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self._model.generate,
                    **inputs,
                    max_length=self.config.response_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=True
                )
                
            response = self.model_setup.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except asyncio.TimeoutError:
            logger.error("Response generation timed out")
            raise TimeoutError("Response generation timed out")
        except Exception as e:
            logger.exception(LogTemplates.MODEL_ERROR.format(error=str(e)))
            raise

    async def stream_response_async(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response asynchronously.

        Args:
            prompt: The input prompt
            config: Optional configuration overrides

        Yields:
            Response text chunks

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If prompt is invalid
            TimeoutError: If streaming times out
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Model is not loaded")

        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")

        try:
            # Validate prompt length
            if len(prompt) > self.config.max_tokens:
                raise ValueError("Prompt exceeds maximum length")

            # Stream response using H2O-accelerated model
            inputs = self._model.prepare_inputs_for_generation(
                input_ids=torch.tensor([self.model_setup.tokenizer.encode(prompt)]),
                attention_mask=None,
                use_cache=True
            )

            with torch.no_grad():
                for outputs in await asyncio.to_thread(
                    self._model.generate,
                    **inputs,
                    max_length=self.config.response_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=True,
                    streaming=True
                ):
                    try:
                        chunk = self.model_setup.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        yield chunk
                    except asyncio.CancelledError:
                        logger.info("Response streaming cancelled")
                        break

        except asyncio.TimeoutError:
            logger.error("Response streaming timed out")
            raise TimeoutError("Response streaming timed out")
        except Exception as e:
            logger.exception(LogTemplates.MODEL_ERROR.format(error=str(e)))
            raise

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update model configuration.

        Args:
            config: New configuration parameters
        """
        self.config = ModelConfig(**{**vars(self.config), **config})

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.

        Returns:
            Dictionary with model information
        """
        model_info = await self.model_manager.get_model_info(self.model_name)
        return {
            "model_path": str(self.model_path),
            "is_loaded": self._is_loaded,
            "config": vars(self.config),
            "model_info": model_info,
            "h2o_config": {
                "window_length": self.config.h2o_window_length,
                "heavy_hitter_tokens": self.config.h2o_heavy_hitter_tokens
            }
        }


def create_model(
    model_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None,
    model_name: str = "default"
) -> ModelHandler:
    """Factory function to create a model instance.

    Args:
        model_path: Optional path to model files
        config: Optional configuration dictionary
        model_name: Name of model to create

    Returns:
        Configured ModelHandler instance
    """
    return ModelHandler(model_path, config, model_name)
