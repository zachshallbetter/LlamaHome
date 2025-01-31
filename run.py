#!/usr/bin/env python3
"""CLI entry point for LlamaHome."""

import asyncio
import sys
from pathlib import Path
from typing import NoReturn

from src.core.resource import GPUConfig, ResourceManager
from src.core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


async def main() -> None:
    """Main entry point."""
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))

        # Initialize logging
        logger.info("Starting LlamaHome...")

        # Setup resource management
        gpu_config = GPUConfig()
        resource_manager = ResourceManager(gpu_config)

        # TODO: Add CLI argument parsing and command execution
        async with resource_manager.optimize():
            # TODO: Run main application logic
            pass

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)


def run() -> NoReturn:
    """Run the application."""
    asyncio.run(main())
    sys.exit(0)


if __name__ == "__main__":
    run()
