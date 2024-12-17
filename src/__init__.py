"""LlamaHome core module."""

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console

from .core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()

def initialize_core() -> None:
    """Initialize core components."""
    try:
        # Create necessary directories
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        models_dir = data_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        cache_dir = data_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        logger.info("Core initialization complete")
        
    except Exception as e:
        logger.error(f"Core initialization error: {e}")
        raise

__all__ = ["initialize_core"]
