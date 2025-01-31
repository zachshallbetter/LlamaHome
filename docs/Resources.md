# Resource Management Guide

## 

## Overview

LlamaHome implements comprehensive resource management for efficient utilization of system resources including memory, GPU, CPU, and I/O operations.

## Resource Configuration

### Core Settings
```yaml
resources:
  memory:
    gpu_reserved: 1024  # MB
    cpu_reserved: 2048  # MB
    swap_threshold: 0.8
    cleanup_threshold: 0.9
  
  gpu:
    memory_fraction: 0.9
    priority_devices: [0, 1]  # Prioritize first two GPUs
    enable_peer_access: true
    balance_load: true
  
  cpu:
    usage_threshold: 0.8
    thread_count: "auto"
    priority: "normal"
  
  io:
    queue_size: 1000
    max_workers: 4
    timeout: 30  # seconds
```

## Resource Management

### Memory Management
```python
from llamahome.resources import MemoryManager

async def manage_memory():
    manager = MemoryManager()
    
    # Get current memory state
    stats = await manager.get_memory_stats()
    print(f"GPU Memory: {stats['gpu_used']}/{stats['gpu_total']} MB")
    print(f"CPU Memory: {stats['cpu_used']}/{stats['cpu_total']} MB")
    
    # Optimize memory usage
    async with manager.optimize():
        # Run memory-intensive operations
        pass
```

### GPU Management
```python
from llamahome.resources import GPUManager

async def manage_gpu():
    manager = GPUManager()
    
    # Get GPU information
    devices = await manager.get_available_devices()
    for device in devices:
        props = await manager.get_device_properties(device)
        print(f"GPU {device}: {props['name']} ({props['memory_total']} MB)")
    
    # Distribute workload
    async with manager.distribute() as devices:
        for device in devices:
            # Device-specific operations
            stats = await manager.get_device_stats(device)
            print(f"Device {device} utilization: {stats['utilization']}%")
```

### Resource Monitoring
```python
from llamahome.resources import ResourceMonitor

async def monitor_resources():
    monitor = ResourceMonitor()
    
    # Start monitoring
    async with monitor:
        # Get current metrics
        metrics = await monitor.get_metrics()
        print(f"GPU Usage: {metrics['gpu_usage']:.2%}")
        print(f"Memory Usage: {metrics['memory_usage']:.2%}")
        print(f"CPU Usage: {metrics['cpu_usage']:.2%}")
        
        # Get alerts
        alerts = await monitor.get_alerts()
        for alert in alerts:
            print(f"Alert: {alert['message']} ({alert['severity']})")
```

## Integration Examples

### Training Integration
```python
from llamahome.training import TrainingPipeline
from llamahome.resources import ResourceManager

async def train_with_resources():
    resource_manager = ResourceManager()
    
    async with resource_manager.optimize() as resources:
        pipeline = TrainingPipeline(
            model,
            device_map=resources.device_map,
            memory_config=resources.memory_config
        )
        await pipeline.train()
```

### Inference Integration
```python
from llamahome.inference import InferencePipeline
from llamahome.resources import ResourceManager

async def inference_with_resources():
    resource_manager = ResourceManager()
    
    async with resource_manager.optimize() as resources:
        pipeline = InferencePipeline(
            model,
            device=resources.optimal_device,
            batch_size=resources.optimal_batch_size
        )
        response = await pipeline.generate("prompt")
```

## Best Practices

1. **Memory Management**
   - Always use context managers
   - Monitor memory usage
   - Implement cleanup handlers
   - Use appropriate batch sizes

2. **GPU Management**
   - Balance workloads across GPUs
   - Monitor GPU memory usage
   - Enable peer access when appropriate
   - Use mixed precision when possible

3. **Resource Monitoring**
   - Set appropriate alerts
   - Monitor resource trends
   - Log resource usage
   - Implement auto-scaling

## Troubleshooting

For resource-related issues, refer to the [Troubleshooting Guide](Troubleshooting.md) for:
- Memory management issues
- GPU allocation problems
- CPU bottlenecks
- I/O performance issues

## Related Documentation
- [Performance Guide](Performance.md) - Performance optimization
- [Monitoring Guide](Monitoring.md) - Resource monitoring
- [Configuration Guide](Config.md) - Resource configuration
- [Cache Guide](Cache.md) - Cache resource management
