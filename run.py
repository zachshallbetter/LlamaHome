#!/usr/bin/env python3
"""LlamaHome CLI runner."""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src import initialize_system
from src.interfaces.cli import CLIInterface
from src.data import create_storage, create_analyzer

# Suppress unnecessary warnings and logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOGGING_LEVEL"] = "error"

console = Console()

def main():
    """Run the LlamaHome CLI."""
    try:
        # Initialize silently
        initialize_system()
        
        # Create and start interface
        interface = CLIInterface()
        interface.start()
        
    except KeyboardInterrupt:
        console.print("\n[green]Goodbye! Thank you for using LlamaHome.[/green]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
