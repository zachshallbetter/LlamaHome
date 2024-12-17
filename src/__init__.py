"""LlamaHome core package."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from llama_recipes import LlamaForCausalLM, LlamaTokenizer

from .utils.log_manager import LogManager, LogTemplates

# Configure logging
logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()

def initialize_system() -> None:
    """Initialize the LlamaHome system."""
    try:
        # Create necessary directories
        Path("data/models").mkdir(parents=True, exist_ok=True)
        Path(".config").mkdir(parents=True, exist_ok=True)
        Path(".logs").mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        from .core import initialize_core
        initialize_core()
        
    except Exception as e:
        logger.error(f"System error during initialization: {e}")
        console.print(f"[red]Error during initialization: {e}[/red]")
        sys.exit(1)

# Import interfaces after system initialization
from .interfaces.cli import CLIInterface

__all__ = [
    # Functions
    "initialize_system",
    
    # Classes
    "CLIInterface",
]
