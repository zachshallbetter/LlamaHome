"""Logging configuration and management."""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional


class LogTemplates(Enum):
    """Common logging templates."""
    
    SYSTEM_STARTUP = auto()
    MODEL_INIT = auto()
    TRAINING = auto()
    INFERENCE = auto()
    BENCHMARK = auto()
    CACHE = auto()


@dataclass
class LogConfig:
    """Logging configuration."""
    
    level: int = logging.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[Path] = None


class LogManager:
    """Manages logging configuration and setup."""

    def __init__(self, template: LogTemplates, config: Optional[LogConfig] = None):
        """Initialize log manager.
        
        Args:
            template: Logging template to use
            config: Optional custom configuration
        """
        self.template = template
        self.config = config or LogConfig()
        self._configure_logging()

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.config.level)
        return logger

    def _configure_logging(self) -> None:
        """Configure logging based on template and config."""
        formatter = logging.Formatter(self.config.format)

        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.config.level)

        # Configure file handler if specified
        if self.config.file:
            file_handler = logging.FileHandler(self.config.file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.config.level)

        # Set root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.level)
        root_logger.addHandler(console_handler)
        if self.config.file:
            root_logger.addHandler(file_handler)