"""Model management implementation."""

from pathlib import Path
from typing import TypeVar

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel

from .config import ModelConfig

T = TypeVar("T", bound="BaseModel")


class ModelManager:
    """Manages model loading and configuration."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model: PreTrainedModel | None = None

    async def load_model(self) -> PreTrainedModel:
        """Load model based on configuration.

        Returns:
            Loaded model

        Raises:
            ValueError: If model loading fails
        """
        try:
            # Load model based on source
            if self.config.model.model_path:
                model = await self._load_local_model()
            else:
                model = await self._load_remote_model()

            # Move to device if specified
            if self.config.resources.device_map == "auto" and torch.cuda.is_available():
                model = model.to("cuda")

            self._model = model
            return model

        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}") from e

    async def _load_local_model(self) -> PreTrainedModel:
        """Load model from local path."""
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.config.model.model_path,
            torch_dtype=self._get_torch_dtype(),
            device_map=self.config.resources.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            revision=self.config.model.model_revision,
            **self._get_quantization_kwargs(),
        )

    async def _load_remote_model(self) -> PreTrainedModel:
        """Load model from remote source."""
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=self._get_torch_dtype(),
            device_map=self.config.resources.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            use_auth_token=self.config.model.use_auth_token,
            revision=self.config.model.model_revision,
            **self._get_quantization_kwargs(),
        )

    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from configuration."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(str(self.config.model.torch_dtype), torch.float16)

    def _get_quantization_kwargs(self) -> dict[str, bool]:
        """Get quantization keyword arguments."""
        if self.config.model.quantization == "int8":
            return {"load_in_8bit": True}
        elif self.config.model.quantization == "int4":
            return {"load_in_4bit": True}
        return {}

    async def save_model(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Save path

        Raises:
            ValueError: If model is not loaded or save fails
        """
        if not self._model:
            raise ValueError("No model loaded")

        try:
            self._model.save_pretrained(path)
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}") from e

    @property
    def model(self) -> PreTrainedModel | None:
        """Get loaded model."""
        return self._model
