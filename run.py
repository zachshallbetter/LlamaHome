"""
Main entry point for LlamaHome application.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.interfaces.cli import cli
from utils.setup_env import setup_environment
from utils.setup_model import setup_model
from utils.system_check import run_system_checks

console = Console()

def initialize_environment():
    """Initialize application environment."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Initializing environment...", total=3)
        
        # Run system checks
        check_results = run_system_checks()
        if check_results["status"] != "ok":
            console.print("[red]System check failed![/red]")
            sys.exit(1)
        progress.advance(task)
        
        # Setup environment
        env_result = setup_environment()
        if not env_result["success"]:
            console.print(f"[red]Environment setup failed: {env_result['error']}[/red]")
            sys.exit(1)
        progress.advance(task)
        
        # Verify directories
        required_dirs = [
            "data",
            "models",
            "config",
            "cache",
            "logs"
        ]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
        progress.advance(task)

def run_application(mode: str = "cli", **kwargs):
    """Run the application in specified mode."""
    try:
        if mode == "cli":
            cli()
        else:
            console.print(f"[red]Unknown mode: {mode}[/red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Application terminated by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Application error: {e}[/red]")
        sys.exit(1)

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["cli"]),
    default="cli",
    help="Application mode"
)
@click.option(
    "--model",
    help="Model to use",
    default=None
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Configuration file",
    default=None
)
def main(mode: str, model: Optional[str], config: Optional[str]):
    """LlamaHome main entry point."""
    try:
        # Initialize environment
        initialize_environment()
        
        # Setup model if specified
        if model:
            setup_result = setup_model(model, config_path=config)
            if not setup_result["success"]:
                console.print(f"[red]Model setup failed: {setup_result['error']}[/red]")
                sys.exit(1)
        
        # Run application
        run_application(mode=mode)
        
    except Exception as e:
        console.print(f"[red]Startup error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
