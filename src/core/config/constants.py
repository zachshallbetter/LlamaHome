"""Constants for configuration management."""

import os
from typing import Final

# Base directories
ROOT_DIR: Final[str] = os.getenv("LLAMAHOME_ROOT", os.path.expanduser("~/.llamahome"))
CONFIG_DIR: Final[str] = os.path.join(ROOT_DIR, "config")
LOCAL_CONFIG_DIR: Final[str] = os.path.join(ROOT_DIR, "config", "local")

# Data directories
DATA_DIR: Final[str] = os.path.join(ROOT_DIR, ".data")
LOCAL_DATA_DIR: Final[str] = os.path.join(DATA_DIR, "local")
CACHE_DIR: Final[str] = os.path.join(ROOT_DIR, ".cache")

# Model directories
MODELS_DIR: Final[str] = os.path.join(DATA_DIR, "models")
DATASETS_DIR: Final[str] = os.path.join(DATA_DIR, "datasets")
EMBEDDINGS_DIR: Final[str] = os.path.join(DATA_DIR, "embeddings")
CHECKPOINTS_DIR: Final[str] = os.path.join(DATA_DIR, "checkpoints")
ARTIFACTS_DIR: Final[str] = os.path.join(DATA_DIR, "artifacts")
MEMORY_DIR: Final[str] = os.path.join(DATA_DIR, "memory")
METRICS_DIR: Final[str] = os.path.join(DATA_DIR, "metrics")
TELEMETRY_DIR: Final[str] = os.path.join(DATA_DIR, "telemetry")
TRAINING_DIR: Final[str] = os.path.join(DATA_DIR, "training")

# Logging
LOG_DIR: Final[str] = os.path.join(ROOT_DIR, "logs")
LOG_LEVEL: Final[str] = os.getenv("LLAMAHOME_LOG_LEVEL", "INFO")

# Training
MAX_SEQUENCE_LENGTH: Final[int] = 2048
BATCH_SIZE: Final[int] = 32
LEARNING_RATE: Final[float] = 5e-5

# Ensure all directories exist
for dir_path in [
    DATA_DIR,
    LOCAL_DATA_DIR,
    MODELS_DIR,
    DATASETS_DIR,
    EMBEDDINGS_DIR,
    CHECKPOINTS_DIR,
    ARTIFACTS_DIR,
    MEMORY_DIR,
    METRICS_DIR,
    TELEMETRY_DIR,
    TRAINING_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)
