# Training System

## Overview

LlamaHome's training system is designed for efficient fine-tuning of large language models using state-of-the-art techniques. The system supports various training approaches including distributed training, LoRA (Low-Rank Adaptation), and full fine-tuning.

## Quick Start

### Basic Training

```bash
# Start training with default configuration
make train DATA_PATH=data/training/dataset.jsonl

# Resume from checkpoint
make train-resume CHECKPOINT=output/training/checkpoint-1000

# Evaluate model
make train-eval MODEL=output/training/final DATA=data/eval/test.jsonl
```

### Distributed Training

```bash
# Single-node multi-GPU training
make train-distributed EPOCHS=10 WORLD_SIZE=4

# Multi-node training
make train-multi-node EPOCHS=10 NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100
```

## Directory Structure

```text
data/
├── training/           # Training datasets
│   ├── raw/           # Original data
│   └── processed/     # Preprocessed data
├── models/            # Model files
│   ├── base/          # Base models
│   └── fine-tuned/    # Fine-tuned models
└── metrics/           # Training metrics
    ├── logs/          # Training logs
    └── tensorboard/   # TensorBoard data
```

## Configuration

Training configuration is managed through YAML files in the `.config` directory:

### Basic Training Config

```yaml
# .config/training_config.yaml
training:
  batch_size: 32
  learning_rate: 5.0e-5
  epochs: 10
  gradient_checkpointing: true
  mixed_precision: true
```

### Distributed Training Config

```yaml
# .config/distributed_config.yaml
distributed:
  backend: "nccl"
  world_size: 4
  num_nodes: 1
  sync_batch_norm: true
```

## Training Pipeline

### 1. Data Processing

- Streaming dataset implementation
- Dynamic batching
- Memory-efficient loading
- Distributed sampling

### 2. Model Preparation

- Model initialization
- Distributed wrapping
- Optimizer setup
- Learning rate scheduling

### 3. Training Loop

- Gradient synchronization
- Metric collection
- Checkpoint management
- Error recovery

## Advanced Features

### 1. Memory Optimization

- Gradient checkpointing
- Mixed precision training
- Memory-efficient attention
- Dynamic batch sizing

### 2. Distributed Training

- Multi-GPU support
- Multi-node capability
- Efficient communication
- Resource management

### 3. Monitoring

- Real-time metrics
- Resource tracking
- Performance visualization
- TensorBoard integration

## Best Practices

### 1. Data Preparation

- Clean and validate data
- Use appropriate batch sizes
- Enable streaming for large datasets
- Implement proper cleanup

### 2. Resource Management

- Monitor GPU memory
- Track CPU utilization
- Optimize I/O operations
- Handle cleanup properly

### 3. Error Handling

- Implement recovery mechanisms
- Save frequent checkpoints
- Monitor resource usage
- Log important metrics

## Troubleshooting

### Common Issues

1. Out of Memory
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision
   - Monitor memory usage

2. Performance Issues
   - Check data loading
   - Optimize batch size
   - Monitor GPU utilization
   - Verify network speed (distributed)

3. Training Instability
   - Adjust learning rate
   - Check gradient norms
   - Validate input data
   - Monitor loss curves

## CLI Commands

```bash
# Basic training
llamahome train <data_path> [options]

# Distributed training
llamahome train-distributed <data_path> [options]

# Training options
--model MODEL         # Base model to use
--output OUTPUT       # Output directory
--config CONFIG       # Custom config file
--world-size SIZE    # Number of processes
--num-nodes NODES    # Number of nodes
```

## Next Steps

1. [Advanced Training](docs/Advanced.md)
2. [Distributed Setup](docs/Distributed.md)
3. [Performance Tuning](docs/Performance.md)
4. [Monitoring Guide](docs/Monitoring.md)