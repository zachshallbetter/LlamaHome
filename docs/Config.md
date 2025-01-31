# Configuration System

LlamaHome uses a comprehensive configuration system based on TOML files and Pydantic models for validation. The system provides type safety, validation, and flexibility while maintaining ease of use.

## Configuration Structure

The configuration system is organized into several components:

```
config/
├── schemas/                    # Configuration schemas
│   ├── model_config.toml      # Model configuration schema
│   ├── training_config.toml   # Training configuration schema
│   ├── inference_config.toml  # Inference configuration schema
│   ├── resource_config.toml   # Resource configuration schema
│   ├── metrics_config.toml    # Metrics configuration schema
│   └── monitoring_config.toml # Monitoring configuration schema
├── model_config.toml          # Default model configuration
├── training_config.toml       # Default training configuration
├── inference_config.toml      # Default inference configuration
├── resource_config.toml       # Default resource configuration
├── metrics_config.toml        # Default metrics configuration
└── monitoring_config.toml     # Default monitoring configuration
```

## Configuration Components

### Model Configuration
Controls model-specific settings:
```toml
[model]
name = "llama"
family = "llama"
size = "13b"
variant = "chat"
revision = "main"
quantization = "none"

[resources]
min_gpu_memory = 16
max_batch_size = 32
max_sequence_length = 32768
```

### Resource Configuration
Manages system resources:
```toml
[gpu]
memory_fraction = 0.9
allow_growth = true
per_process_memory = "12GB"
enable_tf32 = true

[monitor]
check_interval = 1.0
memory_threshold = 0.9
cpu_threshold = 0.8
```

### Metrics Configuration
Controls metrics collection:
```toml
[storage]
storage_type = "local"
retention_days = 30
compression = true
export_format = "parquet"

[metrics]
enabled_metrics = ["cpu", "memory", "gpu", "throughput"]
aggregation_interval = 60
```

### Monitoring Configuration
Manages system monitoring:
```toml
[logging]
log_interval = 60
save_interval = 600
log_level = "INFO"

[visualization]
tensorboard = true
progress_bars = true
plot_metrics = true

[alerts]
enabled = true
alert_on_error = true
notification_backend = "console"
```

## Configuration Loading

Configurations can be loaded in several ways:

1. **From Files**:
```python
from src.core.models.config import ModelConfig

# Load from default location
config = await ModelConfig.load()

# Load from specific directory
config = await ModelConfig.load("path/to/config")
```

2. **From Environment**:
Environment variables can override configuration values. Variables should be prefixed with `LLAMAHOME_`:
```bash
export LLAMAHOME_MODEL_NAME="llama"
export LLAMAHOME_MODEL_SIZE="13b"
export LLAMAHOME_GPU_MEMORY_FRACTION="0.8"
```

3. **Programmatically**:
```python
config = ModelConfig(
    model=ModelSpecs(
        name="llama",
        family="llama",
        size="13b"
    ),
    resources=ResourceSpecs(
        min_gpu_memory=16,
        max_batch_size=32
    )
)
```

## Configuration Validation

All configurations are validated using Pydantic models and schemas:

1. **Type Validation**:
```python
class ResourceSpecs(BaseConfig):
    min_gpu_memory: int = Field(8, ge=8)
    max_batch_size: int = Field(32, ge=1)
    max_sequence_length: int = Field(32768, ge=512)
```

2. **Value Constraints**:
```python
class MonitorConfig(BaseConfig):
    check_interval: float = Field(1.0, gt=0.0)
    memory_threshold: float = Field(0.9, ge=0.0, le=1.0)
    cpu_threshold: float = Field(0.8, ge=0.0, le=1.0)
```

3. **Schema Validation**:
```toml
# In schema file
[gpu]
memory_fraction = { type = "float", min = 0.0, max = 1.0, default = 0.9 }
allow_growth = { type = "bool", default = true }
```

## Best Practices

1. **Configuration Organization**:
   - Keep related settings together in appropriate sections
   - Use descriptive names for configuration keys
   - Include comments for non-obvious settings

2. **Environment Variables**:
   - Use environment variables for sensitive information
   - Use environment variables for deployment-specific settings
   - Follow the `LLAMAHOME_` prefix convention

3. **Validation**:
   - Always define constraints for numeric values
   - Use enums or literals for fixed choices
   - Include meaningful error messages

4. **Documentation**:
   - Document all configuration options
   - Include example configurations
   - Explain the impact of each setting

## Configuration Updates

Configurations can be updated at runtime:

```python
# Update specific values
await config_manager.update_config("resources", {
    "gpu_memory_fraction": 0.8
})

# Merge configurations
await config_manager.merge_configs("model", other_config)
```

## Error Handling

The configuration system provides detailed error messages:

```python
try:
    config = await ModelConfig.load()
except ConfigError as e:
    print(f"Configuration error: {e}")
    # Handle error appropriately
```

## Adding New Configurations

To add a new configuration component:

1. Create a schema file in `config/schemas/`
2. Create a default configuration file in `config/`
3. Define Pydantic models for validation
4. Add loading and validation logic
5. Add tests for the new configuration

Example:
```python
class NewConfig(BaseConfig):
    """New configuration component."""
    setting_1: str
    setting_2: int = Field(42, ge=0)
    
    @classmethod
    async def load(cls, config_dir: str = "config") -> 'NewConfig':
        from ..config.manager import ConfigManager
        manager = ConfigManager(config_dir)
        return await manager.load_config(cls, "new", "new_config.toml")
```
