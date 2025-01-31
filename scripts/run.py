#!/usr/bin/env python3
"""
Main entry point for LlamaHome.

This script handles:
1. Environment setup
2. Configuration loading
3. Pipeline initialization
4. Training/inference execution
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ConfigManager
from src.core.pipeline import Pipeline
from src.core.utils.log_manager import LogManager, LogTemplates


def main():
    """Initialize and run the system."""
    # Initialize logging
    logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
    logger.info("Starting LlamaHome...")

    try:
        # Load configuration
        config = ConfigManager().load_config()
        logger.info("Configuration loaded successfully")

        # Initialize pipeline
        pipeline = Pipeline(config)
        logger.info("Pipeline initialized")

        # Run pipeline
        pipeline.run()
        logger.info("Pipeline execution completed")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

    logger.info("LlamaHome execution completed successfully")


if __name__ == "__main__":
    main() 