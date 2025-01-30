"""Base model interface and abstract classes."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from ..utils.log_manager import LogManager, LogTemplates


class BaseModel(ABC):
    """Base class for all models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize base model.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = LogManager(LogTemplates.MODEL_INIT).get_logger(__name__)
        self.config = config or {}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model."""
        raise NotImplementedError

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the model."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model.

        Args:
            path: Path to save the model to
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model.

        Args:
            path: Path to load the model from
        """
        raise NotImplementedError


# Rest of the file remains unchanged
