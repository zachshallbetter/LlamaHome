"""Model setup and initialization utilities."""

import platform
from pathlib import Path
from typing import Optional

import torch

from ..utils import LogManager, LogTemplates


class ModelSetup:
    """Handles model setup and initialization."""

    def __init__(self):
        """Initialize model setup."""
        self.logger = LogManager(LogTemplates.MODEL_INIT).get_logger(__name__)
        self.device = self._get_device()

    def setup_model(self, model_name: str, version: str, device: Optional[str] = None) -> bool:
        """Set up a model for use.
        
        Args:
            model_name: Name of the model
            version: Model version
            device: Optional device override
            
        Returns:
            True if setup successful
        """
        try:
            device = device or self.device
            self.logger.info(f"Setting up {model_name} version {version} on {device}")
            
            # Add model setup logic here
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            return False

    def _get_device(self) -> str:
        """Get the best available compute device.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
            
        if (platform.system() == "Darwin" and
            platform.machine() == "arm64" and
            hasattr(torch.backends, "mps") and
            torch.backends.mps.is_available()):
            return "mps"
            
        return "cpu"