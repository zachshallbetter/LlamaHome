# LlamaHome Model Configuration and Integration

This document explains how to configure, integrate, and utilize models within LlamaHome.

## Overview

The model integration system provides a robust framework for managing and utilizing language models with the following key features:

- Centralized model configuration management
- LoRA fine-tuning support
- Efficient model loading and caching
- Progress tracking and metrics collection
- Comprehensive error handling
- Async operations support

## Prerequisites

Required Environment:

- Python 3.11 (3.12 and 3.13 not supported due to PyTorch compatibility)
- Poetry for dependency management
- For GPU support:
  - NVIDIA CUDA toolkit 12.1 or higher
  - NVIDIA drivers 525 or higher
  - Minimum GPU memory:
    - 8GB for 3.3-7b model
    - 16GB for 3.3-13b model
    - 80GB for 3.3-70b model

## Model Configuration

### Model Types

Configuration is managed through `.config/models.json`:

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
    }
  }
}
```

### Training Configuration

Training settings are managed through `.config/training_config.yaml`:

```yaml
training:
  # General settings
  batch_size: 4
  max_workers: 4
  max_length: 512
  validation_split: 0.1

  # LoRA settings
  lora:
    r: 8
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
    bias: "none"
```

## Model Management

### ModelManager

The core component for managing models:

```python
class ModelManager:
    """Manages model downloading and organization."""
    
    def __init__(self):
        self.workspace_root = Path.cwd()
        self.models_dir = self.workspace_root / "data/models"
        self.config_file = self.workspace_root / ".config/model_config.yaml"
```

### Directory Structure

```text
data/
├── models/
│   ├── llama/
│   │   ├── 3.3-7b/
│   │   ├── 3.3-13b/
│   │   └── 3.3-70b/
│   └── finetuned/
│       └── llama_3.3-7b_finetuned/
└── training/
    ├── samples/
    ├── processed/
    └── checkpoints/
```

## Model Operations

### Downloading Models

```python
success = model_manager.download_model(
    model_name="llama",
    version="3.3-7b"
)
```

### Model Validation

```python
is_valid = model_manager.validate_model_files(
    model_name="llama",
    version="3.3-7b"
)
```

### Model Removal

```python
model_manager.cleanup_model_files(
    model_name="llama",
    version="3.3-7b"
)
```

## Training Integration

### LoRA Fine-tuning

1. Configuration:
   ```yaml
   lora:
     r: 16
     alpha: 32
     dropout: 0.1
     target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
   ```

2. Training:
   ```python
   await training_manager.train_model(
       model_name="llama",
       model_version="3.3-7b",
       num_epochs=3,
       learning_rate=5e-5
   )
   ```

3. Metrics:
   ```json
   {
     "train_loss": 1.234,
     "eval_loss": 1.123,
     "learning_rate": 5e-5,
     "epoch": 1
   }
   ```

### Early Stopping

Configuration:
```yaml
early_stopping:
  enabled: true
  patience: 3
  min_delta: 0.01
```

## Environment Variables

Key environment variables:

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

# Training Settings
LLAMAHOME_BATCH_SIZE=1000
LLAMAHOME_MAX_WORKERS=4
LLAMAHOME_CACHE_SIZE=1024
```

## Best Practices

### Model Selection

1. Resource Considerations:
   - Check GPU memory requirements
   - Consider model size vs. performance
   - Evaluate quantization options
   - Monitor resource usage

2. Training Setup:
   - Use appropriate batch sizes
   - Enable mixed precision training
   - Configure gradient accumulation
   - Monitor validation metrics

### Error Handling

1. Download Errors:
   - Network connectivity
   - Disk space
   - Checksum verification
   - Retry mechanisms

2. Training Errors:
   - Out of memory
   - Gradient issues
   - Early stopping
   - Checkpoint saving

## Testing

### Test Coverage

1. Model Tests:
   - Download verification
   - File validation
   - Configuration loading
   - Resource cleanup

2. Training Tests:
   - LoRA configuration
   - Progress tracking
   - Metric collection
   - Early stopping

Example test:

```python
@pytest.mark.asyncio
async def test_model_download():
    manager = ModelManager()
    success = await manager.download_model("llama", "3.3-7b")
    assert success
    assert manager.validate_model_files("llama", "3.3-7b")
```

## CLI Integration

The model management system is integrated with the CLI:

```bash
# List available models
llamahome models

# Download a model
llamahome download llama 3.3-7b

# Train a model
llamahome train llama 3.3-7b

# Remove a model
llamahome remove llama 3.3-7b
```

## Performance Optimization

### Memory Management

1. Model Loading:
   - Lazy loading
   - Memory mapping
   - Quantization
   - Device placement

2. Training:
   - Gradient checkpointing
   - Mixed precision
   - Memory monitoring
   - Cache management

### GPU Utilization

1. Training:
   - Batch size optimization
   - Gradient accumulation
   - Multi-GPU support
   - Memory efficiency

2. Inference:
   - Batch processing
   - Stream processing
   - Memory management
   - Resource monitoring
