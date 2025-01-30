"""Constants for configuration management."""

import os
from typing import Final

# Base directories
ROOT_DIR: Final[str] = os.getenv("LLAMAHOME_ROOT", os.path.expanduser("~/.llamahome"))
CONFIG_DIR: Final[str] = os.path.join(ROOT_DIR, "config")
LOCAL_CONFIG_DIR: Final[str] = os.path.join(ROOT_DIR, "config", "local")

# Data directories
DATA_DIR: Final[str] = os.path.join(ROOT_DIR, "data")
LOCAL_DATA_DIR: Final[str] = os.path.join(DATA_DIR, "local")
CACHE_DIR: Final[str] = os.path.join(ROOT_DIR, ".cache")

# Model directories
MODELS_DIR: Final[str] = os.path.join(ROOT_DIR, "models")
CHECKPOINTS_DIR: Final[str] = os.path.join(MODELS_DIR, "checkpoints")

# Logging
LOG_DIR: Final[str] = os.path.join(ROOT_DIR, "logs")
LOG_LEVEL: Final[str] = os.getenv("LLAMAHOME_LOG_LEVEL", "INFO")

# Training
MAX_SEQUENCE_LENGTH: Final[int] = 2048
BATCH_SIZE: Final[int] = 32
LEARNING_RATE: Final[float] = 5e-5
