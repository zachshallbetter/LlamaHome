"""Tests for constants module."""

import os
from pathlib import Path
from typing import Any, Dict

import pytest
import toml

from src.core.model_constants import SUPPORTED_MODELS, ENV_VARS


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create and return a temporary test data directory."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_config_dir(test_data_dir: Path) -> Path:
    """Create and return a temporary config directory."""
    config_dir = test_data_dir / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir


@pytest.fixture(scope="session")
def test_models_dir(test_data_dir: Path) -> Path:
    """Create and return a temporary models directory."""
    models_dir = test_data_dir / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


@pytest.fixture(scope="session")
def base_model_config() -> Dict[str, Any]:
    """Return base model configuration for testing."""
    return {"models": SUPPORTED_MODELS}


@pytest.fixture(scope="function")
def model_config_file(test_config_dir: Path, base_model_config: Dict[str, Any]) -> Path:
    """Create and return a temporary model config file."""
    config_file = test_config_dir / "model_config.toml"
    with open(config_file, "w") as f:
        toml.dump(base_model_config, f)
    return config_file


@pytest.fixture(scope="function")
def clean_env() -> None:
    """Provide a clean environment for testing."""
    # Store original environment
    original_env = dict(os.environ)

    # Clear relevant environment variables
    for var in ENV_VARS:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
