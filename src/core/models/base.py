"""Base model interface and abstract classes."""

from typing import Dict, Optional

from ..utils import LogManager, LogTemplates


class BaseModel:
    """Base class for all models."""


    def __init__(self, config: Optional[Dict] = None):
        """Initialize base model.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = LogManager(LogTemplates.MODEL_INIT).get_logger(__name__)
        self.config = config or {}

# Rest of the file remains unchanged
