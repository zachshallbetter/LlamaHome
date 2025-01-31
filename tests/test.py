"""Test configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Set up test environment."""
    # Create test directories
    test_dirs = [
        "models",
        "cache",
        "training",
        "metrics",
        "logs",
        "local",
        "temp",
        "checkpoints",
    ]
    for d in test_dirs:
        (tmp_path / d).mkdir(parents=True)

    # Set environment variables
    import os

    os.environ["DATA_ROOT"] = str(tmp_path)
    os.environ["CONFIG_DIR"] = str(Path("config"))

    yield

    # Cleanup
    import shutil

    shutil.rmtree(tmp_path)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create temporary config directory.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to temporary config directory
    """
    return tmp_path / "config"
