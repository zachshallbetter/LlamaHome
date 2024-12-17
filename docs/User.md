# LlamaHome User Guide

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Getting Started

### Installation

1. **System Requirements**

   ```text
   - Python 3.11
   - 16GB+ RAM
   - 50GB+ Storage
   - CUDA-capable GPU (recommended)
   ```

2. **Quick Install**

   ```bash
   # Clone repository
   git clone https://github.com/zachshallbetter/llamahome.git
   cd llamahome
   
   # Install dependencies
   make setup
   ```

3. **Initial Configuration**

   ```bash
   # Copy example configuration
   cp .env.example .env
   
   # Edit configuration
   nano .env
   ```

### First Steps

1. **Start CLI**

   ```bash
   make run
   ```

2. **Basic Commands**

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
