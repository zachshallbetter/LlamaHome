# LlamaHome Setup Guide

## Quick Start Guide

### First-Time Setup

1. **System Requirements Check**

   ```bash
   # Check Python version (3.11 or 3.12 required)
   python --version
   
   # Check GPU requirements (if using GPU)
   nvidia-smi  # For NVIDIA GPUs
   ```

2. **Installation**

   ```bash
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -

   # Clone and setup LlamaHome
   git clone https://github.com/zachshallbetter/llamahome.git
   cd llamahome
   make setup
   ```

3. **Initial Configuration**

   ```bash
   # Set up environment
   cp .env.example .env
   # Edit .env with your settings
   ```

### Common Setup Scenarios

1. **CPU-Only Setup**

   ```yaml
   # config/model_config.toml
   compute:
     device: cpu
     threads: 8
     batch_size: 4
   ```

2. **GPU Setup (NVIDIA)**

   ```yaml
   # config/model_config.toml
   compute:
     device: cuda
     gpu_layers: 32
     batch_size: 16
   ```

3. **Apple Silicon Setup**

   ```yaml
   # config/model_config.toml
   compute:
     device: mps
     batch_size: 8
   ```

## Detailed Configuration

### Environment Configuration

1. **Basic Settings**

   ```env
   PYTHON_VERSION=3.11
   LLAMAHOME_ENV=development
   LLAMAHOME_LOG_LEVEL=INFO
   ```

2. **Model Settings**

   ```env
   LLAMA_MODEL=llama3.3
   LLAMA_MODEL_SIZE=13b
   LLAMA_MODEL_VARIANT=chat
   ```

3. **Performance Settings**

   ```env
   LLAMAHOME_BATCH_SIZE=1000
   LLAMAHOME_MAX_WORKERS=4
   LLAMAHOME_CACHE_SIZE=1024
   ```

### Advanced Configuration

1. **H2O Integration**

   ```yaml
   h2o_config:
     enable: true
     window_length: 512
     heavy_hitter_tokens: 128
     position_rolling: true
     max_sequence_length: 32768
   ```

2. **Resource Management**

   ```yaml
   resource_config:
     max_memory: "90%"
     gpu_memory_fraction: 0.8
     cpu_threads: 8
   ```

## Troubleshooting Guide

### Common Issues

1. **Python Version Mismatch**

   ```text
   Problem: "Python 3.13 is not supported"
   Solution: Install Python 3.11 or 3.12
   Command: pyenv install 3.11
   ```

2. **GPU Memory Issues**

   ```text
   Problem: "CUDA out of memory"
   Solutions:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use CPU offloading
   ```

3. **Environment Setup Issues**

   ```text
   Problem: "Environment not configured"
   Solution: 
   1. Check .env file exists
   2. Verify environment variables
   3. Restart application
   ```

### System Compatibility

1. **Operating System Requirements**
   - Linux: Ubuntu 20.04+, CentOS 8+
   - macOS: 12.0+ (Intel/Apple Silicon)
   - Windows: 10/11 with WSL2

2. **GPU Requirements**
   - NVIDIA: CUDA 12.1+, Driver 525+
   - Memory Requirements:
     - 7B model: 8GB VRAM
     - 13B model: 16GB VRAM
     - 70B model: 80GB VRAM

## Performance Optimization

### Memory Management

1. **GPU Memory Optimization**

   ```yaml
   optimization:
     gpu_memory_efficient: true
     gradient_checkpointing: true
     attention_slicing: true
   ```

2. **CPU Memory Optimization**

   ```yaml
   optimization:
     cpu_offload: true
     memory_efficient_attention: true
     pin_memory: true
   ```

### Caching Configuration

1. **Model Cache**

   ```yaml
   cache_config:
     model_cache_size: "10GB"
     cache_format: "fp16"
     eviction_policy: "lru"
   ```

2. **Data Cache**

   ```yaml
   cache_config:
     data_cache_size: "5GB"
     cache_backend: "redis"
     compression: true
   ```

## Maintenance

### Regular Maintenance Tasks

1. **Cache Cleanup**

   ```bash
   make clean-cache
   ```

2. **Update Dependencies**

   ```bash
   make update-deps
   ```

3. **System Check**

   ```bash
   make system-check
   ```

### Backup and Recovery

1. **Configuration Backup**

   ```bash
   make backup-config
   ```

2. **Model Backup**

   ```bash
   make backup-models
   ```

## Advanced Setup

### Custom Integrations

1. **API Integration**

   ```python
   from llamahome.core import LlamaHome
   
   llama = LlamaHome()
   llama.start_server(port=8080)
   ```

2. **Plugin Setup**

   ```python
   from llamahome.plugins import Plugin
   
   class CustomPlugin(Plugin):
       def initialize(self):
           """Plugin initialization."""
           pass
   ```

### Security Configuration

1. **Access Control**

   ```yaml
   security:
     authentication: true
     token_expiry: 3600
     rate_limit: 100
   ```

2. **Encryption**

   ```yaml
   security:
     ssl_enabled: true
     cert_path: "/path/to/cert"
     key_path: "/path/to/key"
   ```

## Role-Based Setup Guide

### Data Scientists

- Focus on model configuration
- Performance optimization
- Data pipeline setup

### DevOps Engineers

- System deployment
- Resource management
- Monitoring setup

### Application Developers

- API integration
- Plugin development
- Custom workflow setup

## Next Steps

1. [Configure Models](docs/Models.md)
2. [Setup Training](docs/Training.md)
3. [API Integration](docs/API.md)
4. [GUI Setup](docs/GUI.md)
