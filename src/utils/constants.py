"""Constants for LlamaHome project."""

from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
LOCAL_DATA_DIR = DATA_DIR / "local"

# Cache directories
CACHE_DIR = ROOT_DIR / ".cache"
PYTEST_CACHE_DIR = CACHE_DIR / "pytest"
TYPE_CHECK_CACHE_DIR = CACHE_DIR / "type_check"

# Config directories
CONFIG_DIR = ROOT_DIR / ".config"
LOCAL_CONFIG_DIR = LOCAL_DATA_DIR / "config"

# Ensure critical directories exist
for directory in [DATA_DIR, LOCAL_DATA_DIR, CACHE_DIR, PYTEST_CACHE_DIR, 
                 TYPE_CHECK_CACHE_DIR, CONFIG_DIR, LOCAL_CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 