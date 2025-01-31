# LlamaHome Configuration System

The LlamaHome configuration system provides a flexible, hierarchical approach to managing settings across the application.

## Directory Structure

```
.
├── .data/              # Root data directory
│   ├── models/         # Model files
│   ├── cache/         # Cache storage
│   ├── training/      # Training data and outputs
│   ├── telemetry/     # System telemetry data
│   ├── memory/        # Memory storage
│   ├── logs/          # Log files
│   ├── local/         # Local model storage
│   └── temp/          # Temporary files
├── config/            # Configuration files
│   ├── model_config.toml       # Model-specific settings
│   ├── training_config.toml    # Training settings
│   ├── distributed_config.toml # Distributed training config
│   ├── system_commands.toml    # System command settings
│   ├── code_check.toml        # Code checking rules
│   └── llamahome.types.ini    # Type checking configuration
└── .env               # Environment variables
```

## Configuration Sources

The configuration system uses multiple sources with the following precedence (highest to lowest):

1. Environment Variables
2. .env File
3. TOML Configuration Files
4. Default Values

## Configuration Classes

### PathConfig
Manages the standard directory structure for data storage.

### ConfigManager
Central configuration manager that loads and provides access to all configuration sources.

### ModelConfig
Handles model-specific configuration including:
- Model paths and cache locations
- Model parameters and settings
- H2O integration settings
- GPU memory requirements
- Model-specific optimizations

## Environment Variables

Key environment variables include:

```bash
# Core Settings
DATA_ROOT=./.data              # Root data directory
LLAMAHOME_MODEL_PATH=./.data/models  # Model storage
MODEL_CACHE_DIR=./.data/cache/models # Model cache

# Model Settings
DEFAULT_MODEL_NAME=llama3.3-13b
USE_LOCAL_MODELS=true
LOCAL_MODELS_PATH=./.data/local

# Llama Settings
LLAMA_MODEL_SIZE=13b
LLAMA_MODEL_VARIANT=chat
LLAMA_MODEL_QUANT=f16
```

## TOML Configuration

### Model Configuration (model_config.toml)
Defines model-specific parameters, requirements, and capabilities.

### Training Configuration (training_config.toml)
Specifies training pipeline settings, optimization parameters, and resource management.

### Distributed Configuration (distributed_config.toml)
Controls distributed training setup, communication, and resource allocation.

## Usage

```python
from src.core.config import ConfigManager, ModelConfig

# Get configuration manager
config_manager = ConfigManager()

# Access model configuration
model_config = ModelConfig()

# Get paths
models_path = config_manager.paths.get("models")
cache_path = config_manager.paths.get("cache")
```

## Extending the Configuration

To add new configuration options:

1. Add settings to appropriate TOML file
2. Update environment variables if needed
3. Create or update corresponding configuration class
4. Update documentation

## Security

- Environment variables take precedence for sensitive settings
- Configuration files should not contain secrets
- Use .env for local development settings
- Production settings should be managed through environment variables
