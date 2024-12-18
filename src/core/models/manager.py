"""Model management and coordination."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from ..utils import LogManager, LogTemplates
from .base import BaseModel


class ModelManager:
    """Manages model loading, saving, and coordination."""


    def __init__(self, config: Optional[Dict] = None):
        """Initialize model manager.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = LogManager(LogTemplates.MODEL_INIT).get_logger(__name__)
        self.config = config or {}
        self.models: Dict[str, BaseModel] = {}


    def load_model(self, model_name: str, model_path: Union[str, Path]) -> BaseModel:
        """Load a model from disk.

        Args:
            model_name: Name of the model
            model_path: Path to model files

        Returns:
            Loaded model instance
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        self.logger.info(f"Loading model {model_name} from {model_path}")
        model = BaseModel()  # Replace with actual model loading
        self.models[model_name] = model
        return model


    def save_model(self, model_name: str, model_path: Union[str, Path]) -> None:
        """Save a model to disk.

        Args:
            model_name: Name of the model to save
            model_path: Path to save model files
        """
        if model_name not in self.models:
            raise KeyError(f"Model not found: {model_name}")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving model {model_name} to {model_path}")
        model = self.models[model_name]
        model.save_pretrained(model_path)


    def list_models(self) -> List[str]:
        """List all loaded models.

        Returns:
            List of model names
        """
        return list(self.models.keys())


    def get_model(self, model_name: str) -> BaseModel:
        """Get a loaded model by name.

        Args:
            model_name: Name of the model to get

        Returns:
            Model instance
        """
        if model_name not in self.models:
            raise KeyError(f"Model not found: {model_name}")
        return self.models[model_name]


    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory.

        Args:
            model_name: Name of the model to unload
        """
        if model_name not in self.models:
            raise KeyError(f"Model not found: {model_name}")
        del self.models[model_name]
        self.logger.info(f"Unloaded model: {model_name}")


    def download_model(self, model_name: str, model_url: str) -> Path:
        """Download a model from a URL.

        Args:
            model_name: Name to give the downloaded model
            model_url: URL to download from

        Returns:
            Path where model was downloaded
        """
        # Add model downloading logic here
        download_path = Path(f"models/{model_name}")
        download_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Downloading model {model_name} from {model_url}")
        return download_path
