# LlamaHome User Guide

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Overview

LlamaHome is a user-friendly interface for interacting with large language models. It provides a command-line interface (CLI) for basic interactions and a graphical user interface (GUI) for more advanced features.

## Getting Started

### Installation

1. **System Requirements**

   ```text
   - Python 3.11
   - 16GB+ RAM
   - 50GB+ Storage
   - CUDA-capable GPU (recommended)
   ```

2. **Install Dependencies**

   ```bash
   # Install Python 3.11
   # On Ubuntu/Debian:
   sudo apt update
   sudo apt install python3.11 python3.11-venv

   # On macOS with Homebrew:
   brew install python@3.11

   # Install Trunk CLI
   curl https://get.trunk.io -fsSL | bash

   # Install Git
   # Ubuntu/Debian:
   sudo apt install git
   # macOS:
   brew install git
   ```

3. **Clone and Setup**

   ```bash
   # Clone repository
   git clone https://github.com/zachshallbetter/llamahome.git
   cd llamahome

   # Run installation script
   python3.11 scripts/install.py
   ```

### GPU Support

GPU acceleration support depends on your platform:

#### Apple Silicon (M1/M2) Macs
M1/M2 Macs use Metal Performance Shaders (MPS) for GPU acceleration:
```bash
# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"

# In your .env file:
DEVICE=mps  # Use Metal GPU acceleration
```

#### NVIDIA GPUs (Linux/Windows)
CUDA acceleration requires:
- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit 11.7+
- Proper CUDA_HOME environment variable setup

To enable CUDA support:

1. Install NVIDIA CUDA Toolkit:
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-cuda-toolkit
   
   # Or download from NVIDIA website
   # https://developer.nvidia.com/cuda-downloads
   ```

2. Set CUDA_HOME environment variable:
   ```bash
   # Add to your .bashrc or .zshrc
   export CUDA_HOME=/usr/local/cuda  # Adjust path as needed
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

3. Verify installation:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.cuda.is_available())"
   ```

On macOS or systems without CUDA, the installation will skip CUDA-specific features and run in CPU-only mode.

### First Run

1. **Configure Environment**

   ```bash
   # Copy and edit configuration
   cp .env.example .env
   nano .env

   # Minimum required settings:
   MODEL_CACHE_DIR=.cache/models
   CUDA_VISIBLE_DEVICES=0  # Set to -1 for CPU only
   ```

2. **Start Application**

   ```bash
   # Activate virtual environment
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows

   # Start CLI interface
   python -m src.interfaces.cli

   # Or use make command
   make run
   ```

3. **Quick Test**

   ```bash
   # In the CLI interface:
   > models  # List available models
   > download llama3.3-7b  # Download base model
   > chat  # Start chat session
   ```

### Basic Commands

```text
help           Show available commands
models         List available models
download       Download a model
chat           Start chat session
quit           Exit application
```

## Basic Usage

### Chat Interface

1. **Start Chat**

   ```bash
   > chat
   Model (llama3.3): 
   ```

2. **Basic Interaction**

   ```text
   You: Hello, how are you?
   Assistant: Hello! I'm functioning well and ready to help.
   
   You: What can you help me with?
   Assistant: I can assist with various tasks...
   ```

3. **Chat Commands**

   ```text
   /help          Show chat commands
   /clear         Clear chat history
   /model         Switch model
   /save          Save conversation
   /exit          Exit chat mode
   ```

### Model Management

1. **List Models**

   ```bash
   > models
   Available models:
   - llama3.3-7b
   - llama3.3-13b
   - llama3.3-70b
   ```

2. **Download Model**

   ```bash
   > download llama3.3-7b
   Downloading model...
   Progress: [====================] 100%
   Model downloaded successfully
   ```

3. **Remove Model**

   ```bash
   > remove llama3.3-7b
   Removing model...
   Model removed successfully
   ```

## Advanced Features

### Configuration

1. **Model Settings**

   ```yaml
   # config/model_config.toml
   models:
     llama3.3:
       optimization:
         quantization: "int8"
         gpu_layers: 32
         batch_size: 16
   ```

2. **System Settings**

   ```yaml
   # config/system_config.toml
   system:
     cache_size: "10GB"
     max_memory: "90%"
     log_level: "INFO"
   ```

3. **Training Settings**

   ```yaml
   # config/training_config.toml
   training:
     batch_size: 32
     learning_rate: 0.001
     epochs: 10
   ```

### Performance Optimization

1. **Memory Usage**

   ```bash
   # Enable memory optimization
   > config set memory.optimization true
   
   # Set cache size
   > config set cache.size 10GB
   ```

2. **GPU Settings**

   ```bash
   # Set GPU layers
   > config set gpu.layers 32
   
   # Enable quantization
   > config set gpu.quantization int8
   ```

3. **Batch Processing**

   ```bash
   # Set batch size
   > config set processing.batch_size 16
   
   # Enable batch optimization
   > config set processing.optimize true
   ```

### Advanced Commands

1. **Training**

   ```bash
   # Start training
   > train --data path/to/data --epochs 10
   
   # Fine-tune model
   > finetune --model llama3.3-7b --data path/to/data
   ```

2. **Evaluation**

   ```bash
   # Evaluate model
   > evaluate --model llama3.3-7b --data path/to/test
   
   # Benchmark performance
   > benchmark --model llama3.3-7b
   ```

3. **Export/Import**

   ```bash
   # Export conversation
   > export chat.json
   
   # Import conversation
   > import chat.json
   ```

## Best Practices

### Resource Management

1. **Memory Management**

   ```text
   - Monitor memory usage
   - Clear cache regularly
   - Use appropriate batch sizes
   - Enable memory optimization
   ```

2. **GPU Optimization**

   ```text
   - Use appropriate quantization
   - Optimize GPU layers
   - Monitor GPU memory
   - Enable H2O optimization
   ```

3. **Storage Management**

   ```text
   - Clean unused models
   - Compress chat logs
   - Archive old data
   - Monitor disk usage
   ```

### Performance Tips

1. **Model Selection**

   ```text
   - Choose appropriate model size
   - Consider hardware limitations
   - Balance speed vs quality
   - Use optimized variants
   ```

2. **Batch Processing**

   ```text
   - Use optimal batch sizes
   - Enable batch optimization
   - Monitor throughput
   - Balance latency
   ```

3. **Cache Usage**

   ```text
   - Enable smart caching
   - Set appropriate cache size
   - Monitor cache hits
   - Clear cache when needed
   ```

## Troubleshooting

### Common Issues

1. **Installation Problems**

   ```text
   Q: Poetry installation fails
   A: Ensure Python 3.11 is installed
   
   Q: Dependency conflicts
   A: Clear Poetry cache and reinstall
   ```

2. **Runtime Issues**

   ```text
   Q: Out of memory
   A: Reduce batch size or enable optimization
   
   Q: Slow performance
   A: Check GPU settings and optimization
   ```

3. **Model Problems**

   ```text
   Q: Model fails to load
   A: Check model files and GPU memory
   
   Q: Poor model output
   A: Adjust temperature and settings
   ```

### Debug Mode

Enable detailed logging:

```bash
# Set debug level
> config set log.level DEBUG

# Enable debug output
> debug on
```

## Examples

1. **Simple Chat**

   ```text
   > chat
   You: Summarize this text: [paste text]
   Assistant: Here's a summary...
   ```

2. **File Processing**

   ```text
   > process file.txt
   Processing file...
   Results saved to output.txt
   ```

3. **Model Switching**

   ```text
   > model llama3.3-13b
   Switching to llama3.3-13b...
   Model ready
   ```

### Advanced Usage

1. **Custom Processing**

   ```python
   from llamahome import LlamaHome
   
   app = LlamaHome()
   result = await app.process(
       text="Process this",
       temperature=0.7,
       max_tokens=100
   )
   ```

2. **Batch Processing**

   ```python
   results = await app.process_batch(
       texts=["Text 1", "Text 2"],
       batch_size=16
   )
   ```

3. **Stream Processing**

   ```python
   async for chunk in app.stream_response(
       "Generate long text"
   ):
       print(chunk, end="", flush=True)
   ```

## Additional Resources

- [API Reference](API.md)
- [Configuration Guide](Config.md)
- [Performance Guide](Performance.md)
- [FAQ](FAQ.md)

## Updates

This guide is regularly updated. For the latest information:

1. Check documentation updates
2. Review changelog
3. Follow announcements
4. Join community discussions

Remember to check the [Documentation](.) directory for more detailed information on specific topics.
