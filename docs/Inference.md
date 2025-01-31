# Inference Guide

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Streaming Responses](#streaming-responses)
- [Batch Processing](#batch-processing)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Common Issues](#common-issues)

## Overview

This document provides a comprehensive guide on how to use LlamaHome's inference capabilities effectively. 

## Basic Usage

The simplest way to run inference is using the `InferencePipeline`:

```python
from src.inference import InferenceConfig, InferencePipeline

config = InferenceConfig(
    processing=ProcessingConfig(
        max_length=512,
        temperature=0.7,
        top_p=0.9
    )
)

pipeline = InferencePipeline("facebook/opt-1.3b", config)
response = await pipeline.generate("What is machine learning?")
```

## Streaming Responses

For real-time responses, use the `StreamingPipeline`:

```python
from src.inference import StreamingPipeline

pipeline = StreamingPipeline("facebook/opt-1.3b", config)
async for chunk in pipeline.generate_stream("Explain quantum computing"):
    print(chunk, end="", flush=True)
```

## Batch Processing

Process multiple prompts efficiently:

```python
prompts = [
    "What is Python?",
    "Explain databases",
    "How does the internet work?"
]

responses = await pipeline.generate_batch(prompts)
```

## Configuration

### Processing Settings

```python
config = InferenceConfig(
    processing=ProcessingConfig(
        max_length=512,      # Maximum response length
        temperature=0.7,     # Randomness (0.0-1.0)
        top_p=0.9,          # Nucleus sampling
        top_k=50,           # Top-k sampling
        num_beams=1,        # Beam search width
        batch_size=1        # Batch size for processing
    )
)
```

### Resource Management

```python
config = InferenceConfig(
    resource=ResourceConfig(
        gpu_memory_fraction=0.9,    # GPU memory limit
        cpu_usage_threshold=0.8,    # CPU usage limit
        max_parallel_requests=10     # Concurrent request limit
    )
)
```

### Cache Configuration

```python
config = InferenceConfig(
    cache=CacheConfig(
        memory_size=1000,    # Memory cache size (MB)
        disk_size=10000,     # Disk cache size (MB)
        use_mmap=True        # Use memory mapping
    )
)
```

## Best Practices

1. **Resource Management**
   - Use context managers for resource optimization
   - Monitor memory usage during heavy inference
   - Clean up resources after use

2. **Performance Optimization**
   - Batch similar requests when possible
   - Use appropriate temperature settings
   - Consider streaming for long responses

3. **Error Handling**
   - Implement proper timeout handling
   - Handle resource exhaustion gracefully
   - Monitor inference performance

4. **Model Settings**
   - Adjust temperature based on task needs
   - Use appropriate max_length settings
   - Consider beam search for quality vs speed

## Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Lower max_length
   - Use resource management

2. **Slow Response Times**
   - Enable batching
   - Optimize model settings
   - Monitor system resources

3. **Quality Issues**
   - Adjust temperature/top_p
   - Consider beam search
   - Review prompt engineering 