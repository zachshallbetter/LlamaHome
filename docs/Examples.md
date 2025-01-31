# Examples

## Table of Contents

- [Overview](#overview)
- [Training Examples](#training-examples)
- [Inference Examples](#inference-examples)
- [Resource Management](#resource-management)
- [Best Practices](#best-practices)

This document provides examples of common usage patterns for LlamaHome.

## Training Examples

### Basic Training
The [basic training example](../examples/train_model.py) demonstrates the training pipeline:
```python
from llamahome import LlamaHome
from llamahome.training import TrainingConfig

config = TrainingConfig(
    model="llama-3.3-13b",
    dataset="custom_dataset",
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3
)

async with LlamaHome(config) as llm:
    await llm.train()
```

### Distributed Training
The [distributed training example](../examples/train_distributed.py) shows multi-GPU training:
```python
from llamahome.training import DistributedConfig

dist_config = DistributedConfig(
    world_size=4,
    backend="nccl",
    init_method="tcp://localhost:23456"
)

async with LlamaHome(config, distributed=dist_config) as llm:
    await llm.train_distributed()
```

### Custom Dataset Training
The [custom dataset example](../examples/train_custom_data.py) demonstrates training with custom data:
```python
from llamahome.data import DatasetConfig, DataCollator

dataset_config = DatasetConfig(
    train_path="path/to/train",
    eval_path="path/to/eval",
    collator=DataCollator(
        max_length=512,
        padding=True
    )
)

async with LlamaHome(config, dataset=dataset_config) as llm:
    await llm.train()
```

### Training with H2O Optimization
```python
from llamahome.h2o import H2OConfig

h2o_config = H2OConfig(
    window_length=512,
    heavy_hitters=128,
    position_rolling=True
)

async with LlamaHome(config, h2o=h2o_config) as llm:
    await llm.train_with_h2o()
```

### Training Monitoring
```python
from llamahome.monitoring import TrainingMonitor

monitor = TrainingMonitor(
    log_dir="./logs",
    save_steps=100,
    eval_steps=50
)

async with LlamaHome(config, monitor=monitor) as llm:
    await llm.train()
    metrics = await monitor.get_training_metrics()
```

## Resource Management

### Memory Management
The [memory management example](../examples/resource_memory.py) demonstrates memory optimization:

```python
from src.core import ResourceConfig, ResourceManager

manager = ResourceManager(config)
with manager.optimize():
    # Memory-intensive operations
    pass
```

### Performance Monitoring
The [monitoring example](../examples/resource_monitor.py) shows how to track system performance:

```python
from src.core import MonitorConfig, PerformanceMonitor

monitor = PerformanceMonitor(config)
await monitor.start()
metrics = await monitor.get_metrics()
await monitor.stop()
```

### Multi-GPU Management
The [multi-GPU example](../examples/resource_multi_gpu.py) demonstrates GPU resource management:

```python
from src.core import MultiGPUManager

manager = MultiGPUManager(config)
async with manager.distribute() as devices:
    for device in devices:
        # Device-specific operations
        print(f"Using device: {device}")
        pass
```

## Inference Examples

### Basic Inference
The [basic inference example](../examples/inference_basic.py) shows how to run inference:

```python
from src.inference import InferenceConfig, InferencePipeline

config = InferenceConfig(
    processing=ProcessingConfig(
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        do_sample=True,
        repetition_penalty=1.1,
        length_penalty=1.0,
        early_stopping=True
    )
)

pipeline = InferencePipeline("facebook/opt-1.3b", config)
response = await pipeline.generate("What is machine learning?")
print(f"Response: {response}")
```

### Streaming Inference
The [streaming example](../examples/inference_stream.py) demonstrates real-time responses:

```python
from src.inference import StreamingPipeline

pipeline = StreamingPipeline("facebook/opt-1.3b", config)
async for chunk in pipeline.generate_stream("Explain quantum computing"):
    print(chunk, end="", flush=True)
```

### Batch Inference
Process multiple prompts efficiently:

```python
prompts = [
    "What is Python?",
    "Explain databases",
    "How does the internet work?"
]

responses = await pipeline.generate_batch(prompts)
for prompt, response in zip(prompts, responses):
    print(f"\nQ: {prompt}\nA: {response}")
```

## Best Practices

1. **Inference**
   - Use streaming for long responses
   - Batch similar requests
   - Monitor resource usage
   - Handle errors gracefully

2. **Resource Management**
   - Always use context managers
   - Monitor memory usage
   - Clean up resources properly

3. **Training**
   - Start with basic training
   - Use appropriate batch sizes
   - Monitor training metrics

4. **Custom Datasets**
   - Validate data properly
   - Use efficient loading
   - Handle edge cases

For more detailed information, refer to:
- [Inference Guide](../docs/Inference.md)
- [Training Guide](../docs/Training.md)
- [Resource Management Guide](../docs/Resources.md)
