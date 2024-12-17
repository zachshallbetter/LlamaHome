# LlamaHome Configuration Guide

This document details the configuration system and available settings in LlamaHome.

## Configuration Files

### Overview

LlamaHome uses multiple configuration files for different aspects of the system:```text
.config/
├── models.json           # Model definitions and versions
├── training_config.yaml  # Training parameters
├── llamahome.types.ini  # Type checking configuration
└── code_check.yaml      # Code quality settings
```

### Environment Variables

Primary configuration through `.env`:

```bash
# Core settings
PYTHONPATH=./src:${PYTHONPATH}
LLAMAHOME_ENV=development

# Llama Settings
LLAMA_MODEL=llama3.3
LLAMA_MODEL_SIZE=13b
LLAMA_MODEL_VARIANT=chat
LLAMA_MODEL_QUANT=f16
LLAMA_NUM_GPU_LAYERS=32
LLAMA_MAX_SEQ_LEN=32768
LLAMA_MAX_BATCH_SIZE=8

# H2O Settings
LLAMA_H2O_ENABLED=true
LLAMA_H2O_WINDOW_LENGTH=512
LLAMA_H2O_HEAVY_HITTERS=128

# Logging
LLAMAHOME_LOG_LEVEL=INFO
LLAMAHOME_LOG_FILE=./.logs/llamahome.log

# Performance
LLAMAHOME_BATCH_SIZE=1000
LLAMAHOME_MAX_WORKERS=4
LLAMAHOME_CACHE_SIZE=1024

# API Settings
LLAMAHOME_API_HOST=localhost
LLAMAHOME_API_PORT=8000
LLAMAHOME_API_TIMEOUT=30

# Document Processing
LLAMAHOME_MAX_FILE_SIZE=10485760  # 10MB
LLAMAHOME_SUPPORTED_FORMATS=txt,md,json,xml,yaml,docx,xlsx,pptx,pdf
```

## Model Configuration

### models.json

Defines available models and their configurations:

```json
{
  "llama": {
    "formats": ["meta"],
    "versions": [
      "3.3-70b",
      "3.2-1b",
      "3.2-3b",
      "3.2-11b",
      "3.2-90b",
      "3.1-8b",
      "3.1-405b"
    ],
    "default_version": "3.3-70b",
    "min_gpu_memory": {
      "3.3-70b": 40,
      "3.2-1b": 4,
      "3.2-3b": 6,
      "3.2-11b": 12,
      "3.2-90b": 48,
      "3.1-8b": 8,
      "3.1-405b": 128
    },
    "urls": {
      "3.3-70b": "https://llama3-3.llamameta.net/*"
    },
    "requires_auth": false
  },
  "gpt4": {
    "formats": ["api"],
    "versions": ["4-turbo", "4"],
    "default_version": "4-turbo",
    "api_required": true,
    "urls": {}
  }
}
```

### training_config.yaml

Training-specific configuration:

```yaml
training:
  # General settings
  batch_size: 4
  max_workers: 4
  max_length: 512
  validation_split: 0.1

  # Training parameters
  epochs: 3
  learning_rate: 5e-5
  warmup_steps: 100
  weight_decay: 0.01
  gradient_accumulation_steps: 4
  fp16: true
  
  # LoRA settings
  lora:
    r: 8
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
    bias: "none"
    
  # Early stopping
  early_stopping:
    enabled: true
    patience: 3
    min_delta: 0.01
    
  # Checkpointing
  checkpointing:
    save_strategy: "epoch"
    save_total_limit: 3
    
  # Logging
  logging:
    steps: 10
    eval_steps: 100
```

## Type Checking Configuration

### llamahome.types.ini

MyPy configuration for type checking:

```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True

[mypy.plugins.torch.*]
init_forbid_untyped_decorators = False

[mypy.plugins.transformers.*]
init_forbid_untyped_decorators = False
```

## Code Quality Configuration

### code_check.yaml

Code quality check settings:

```yaml
execution:
  jobs: 4
  cache: true
  throttle: 1000

reporting:
  summary: true
  details: true
  save-to-file: "code_check_results.txt"

basepath: "."
recursive: true

ignore:
  - "**/__pycache__/**"
  - "**/.mypy_cache/**"
  - "**/.pytest_cache/**"
  - "**/node_modules/**"
  - "**/.venv/**"
```

## Cache Configuration

### Cache Directory Structure

```text
.cache/
├── models/          # Model weights and parameters
│   ├── llama/
│   └── gpt4/
├── training/        # Training artifacts
│   ├── datasets/
│   └── metrics/
├── system/          # System-level cache
│   ├── pytest/
│   └── mypy/
└── pycache/         # Python bytecode cache
```

## Log Configuration

### Log Directory Structure

```text
.logs/
├── app/
│   ├── error/
│   ├── access/
│   └── debug/
├── models/
│   ├── llama/
│   └── gpt4/
└── system/
    ├── setup/
    └── monitor/
```

## Configuration Management

### Loading Configuration

```python
class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self):
        self.workspace_root = Path.cwd()
        self.config_dir = self.workspace_root / ".config"
        self._load_configs()
        
    def _load_configs(self):
        """Load all configuration files."""
        self.model_config = self._load_json("models.json")
        self.training_config = self._load_yaml("training_config.yaml")
```

### Validation Rules

1. Environment Variables:
   - Required variables must be set
   - Values must match expected types
   - Paths must be valid

2. Model Configuration:
   - Model versions must be valid
   - URLs must be accessible
   - GPU requirements must be specified

3. Training Configuration:
   - Parameters must be within valid ranges
   - Paths must exist
   - Resource limits must be reasonable

### Configuration Updates

1. Runtime Updates:
   ```python
   config_manager.update_training_config({
       "batch_size": 8,
       "learning_rate": 1e-4
   })
   ```

2. File Updates:
   ```python
   config_manager.save_training_config(updated_config)
   ```

## Best Practices

### 1. Environment Variables

- Use meaningful prefixes (LLAMAHOME_, LLAMA_)
- Document all variables
- Provide default values
- Validate on startup

### 2. Configuration Files

- Use appropriate formats (JSON, YAML)
- Include comments
- Version control configs
- Validate schema

### 3. Security

- Protect sensitive values
- Use environment for secrets
- Validate all inputs
- Sanitize paths

### 4. Maintenance

- Regular validation
- Clean old configs
- Update documentation
- Monitor usage

## Usage Examples

### 1. Loading Configuration

```python
from utils.config import ConfigManager

config = ConfigManager()
model_config = config.get_model_config("llama")
training_config = config.get_training_config()
```

### 2. Updating Settings

```python
# Update training settings
config.update_training_config({
    "batch_size": 8,
    "lora": {
        "r": 16,
        "dropout": 0.2
    }
})

# Update model settings
config.update_model_config("llama", {
    "default_version": "3.3-13b"
})
```

### 3. Validation

```python
# Validate configuration
if config.validate():
    print("Configuration is valid")
else:
    print("Configuration errors:", config.errors)
```

### 4. Environment Setup

```bash
# Development
export LLAMAHOME_ENV=development
make setup

# Production
export LLAMAHOME_ENV=production
make setup
``` 

