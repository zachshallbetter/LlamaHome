"""Constants for configuration management."""

import os

# Base directories
ROOT_DIR = os.getenv('LLAMAHOME_ROOT', os.path.expanduser('~/.llamahome'))
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
LOCAL_CONFIG_DIR = os.path.join(ROOT_DIR, 'config', 'local')

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOCAL_DATA_DIR = os.path.join(DATA_DIR, 'local')
CACHE_DIR = os.path.join(ROOT_DIR, '.cache')

# Model directories
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')

# Logging
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_LEVEL = os.getenv('LLAMAHOME_LOG_LEVEL', 'INFO')

# Training
MAX_SEQUENCE_LENGTH = 2048
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
