"""LlamaHome package initialization."""

from pathlib import Path
from rich.console import Console
from .core.utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()


def initialize_project() -> None:
    """Initialize project directories and components."""
    try:
        # Create necessary directories
        subdirs = [
            "models",
            "cache",
            "training",
            "evaluation",
            "monitoring",
            "optimization",
            "data",
            "resources",
            "config",
            "benchmark",
        ]
        for subdir in subdirs:
            dir_path = Path("data") / subdir
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logger.info("Project initialization complete")

    except Exception as e:
        logger.error("Project initialization error: %s", str(e))
        raise


__all__ = ["initialize_project"]
