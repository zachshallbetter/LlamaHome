"""
Main entry point for LlamaHome application.
"""

import sys

from rich.console import Console

from src.core.utils import LogManager, LogTemplates
from src.core.utils.system_check import SystemCheck
from src.interfaces.cli import cli

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()


def main() -> None:
    """Main entry point."""
    try:
        # Run system checks
        system_checker = SystemCheck()
        if not system_checker.check_system_requirements():
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
