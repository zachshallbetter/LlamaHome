"""LlamaHome core module."""

from pathlib import Path
from rich.console import Console
from .core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()

def initialize_core() -> None:
    """Initialize core components."""
    try:
        # Create necessary directories
        for subdir in ["models", "cache", "training", "evaluation", "monitoring", "optimization", "data", "resources", "config", "benchmark"]:
            dir_path = Path("data") / subdir
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logger.info("Core initialization complete")

    except Exception as e:
        logger.error(f"Core initialization error: {e}")
        raise

__all__ = ["initialize_core"]
