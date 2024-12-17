"""
Main entry point for LlamaHome application.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
import torch
from rich.console import Console

from src.core.utils import (
    LogManager,
    LogTemplates,
    setup_model,
    system_check
)
from src.interfaces.cli import cli

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()

def main():
    """Main entry point."""
    try:
        # Run system checks
        if not system_check.check_system_requirements():
            console.print("[red]System requirements not met[/red]")
            sys.exit(1)
        
        # Run CLI
        cli()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Fatal error")
        sys.exit(1)

if __name__ == "__main__":
    main()
