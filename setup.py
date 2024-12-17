#!/usr/bin/env python3
"""LlamaHome setup script."""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
import os

from utils.system_check import SystemCheck
from utils.cache_manager import CacheManager
from utils.setup_env import EnvironmentSetup
from utils.setup_model import ModelSetup
from utils.model_manager import ModelManager
from utils.log_manager import LogManager, LogTemplates

# Load environment variables
load_dotenv()

# Suppress unnecessary warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()
logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


def setup_llamahome() -> int:
    """Set up LlamaHome environment.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        console.print("\n[bold]LlamaHome Setup[/bold]\n")

        # Get workspace root
        workspace_root = Path.cwd()

        # System checks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Checking system configuration...")
            
            system_check = SystemCheck()
            system_info = system_check.get_system_info()
            
            progress.update(task, completed=True)
            
            # Display system info
            console.print(Panel.fit(
                "\n".join(f"{k}: {v}" for k, v in system_info.items()),
                title="System Configuration"
            ))

        # Initialize core components silently
        cache_manager = CacheManager()
        env_setup = EnvironmentSetup()
        model_setup = ModelSetup()
        
        # Set up environment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task1 = progress.add_task("Setting up environment...")
            env_setup.setup_environment()
            cache_manager.clean_cache("pycache")
            progress.update(task1, completed=True)
            
        # Set up model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task2 = progress.add_task("Setting up model...")
            # Create model directory
            model_path = workspace_root / "data/models"
            model_path.mkdir(parents=True, exist_ok=True)
            model_setup.setup_model("llama")
            progress.update(task2, completed=True)

        # Display success message
        console.print(Panel.fit(
            "Setup completed successfully!\nRun 'make run' to start LlamaHome",
            title="Success",
            padding=(1, 2)
        ))
        
        return 0

    except Exception as e:
        console.print(f"[red]Error during setup: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(setup_llamahome())
