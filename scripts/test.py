#!/usr/bin/env python3
"""
Test runner for LlamaHome.

This script provides:
1. Test suite execution
2. Coverage reporting
3. Test environment setup
4. Result reporting
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.core.utils.log_manager import LogManager, LogTemplates


def run_tests():
    """Run the test suite with coverage reporting."""
    logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
    logger.info("Starting test suite execution...")

    try:
        # Set test environment variables
        os.environ["TESTING"] = "true"
        
        # Configure test arguments
        args = [
            "--verbose",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
            "tests/"
        ]

        # Run tests
        result = pytest.main(args)
        
        if result == 0:
            logger.info("All tests passed successfully")
        else:
            logger.error(f"Test suite failed with exit code: {result}")
            sys.exit(result)

    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        raise

    finally:
        # Clean up test environment
        os.environ.pop("TESTING", None)


if __name__ == "__main__":
    run_tests() 