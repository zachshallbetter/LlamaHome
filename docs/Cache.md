# Cache System

## Table of Contents

- [Overview](#overview)
- [Cache Configuration](#cache-configuration)
- [Cache Types](#cache-types)
- [Cache Management](#cache-management)
- [Best Practices](#best-practices)

## Overview

LlamaHome's caching system provides efficient data and model caching capabilities with configurable policies and multiple storage backends. The system is designed to optimize memory usage and improve training performance.

## Cache Configuration

### Core Settings

```yaml
cache:
  memory:
    size: 1000  # MB
    type: "lru"
    compression: true
  disk:
    size: 10000  # MB
    path: "./cache"
    cleanup_interval: 3600  # seconds
```

### Cache Types

1. **Model Cache**
   ```yaml
   model_cache:
     size: 1024  # MB
     format: "safetensors"
     compression: true
     cleanup_policy: "lru"
   ```

2. **Training Cache**
   ```yaml
   training_cache:
     size: 512  # MB
     batch_buffer: 32
     prefetch: 4
     cleanup_policy: "fifo"
   ```

3. **System Cache**
   ```yaml
   system_cache:
     size: 256  # MB
     temp_dir: ".cache/temp"
     max_age: 86400  # seconds
     cleanup_policy: "time"
   ```

## Cache Management

### Automatic Cleanup
```python
class CacheManager:
    """Manages cache lifecycle and cleanup."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cleanup_scheduler = AsyncScheduler()
        
    async def start(self):
        """Start cache management tasks."""
        await self.cleanup_scheduler.schedule(
            self.cleanup,
            interval=self.config.cleanup_interval
        )
```

### Integration with Monitoring
```python
from llamahome.monitoring import CacheMetricsCollector
from llamahome.cache import CacheManager

class MonitoredCacheManager(CacheManager):
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.metrics = CacheMetricsCollector()
        
    async def get_metrics(self):
        return {
            "hit_rate": await self.metrics.get_hit_rate(),
            "memory_usage": await self.metrics.get_memory_usage(),
            "eviction_count": await self.metrics.get_eviction_count()
        }
```

### Integration with Resource Management
```python
from llamahome.resources import ResourceManager

async with ResourceManager() as resources:
    cache_size = await resources.get_available_memory() * 0.3  # 30% of available memory
    cache_config = CacheConfig(memory_size=cache_size)
    cache = CacheManager(cache_config)
```

## Best Practices

1. **Memory Management**
   - Monitor cache usage
   - Set appropriate size limits
   - Enable compression when beneficial
   - Regular cleanup scheduling

2. **Performance Optimization**
   - Use appropriate cache types
   - Configure cleanup policies
   - Monitor hit rates
   - Optimize prefetch settings

3. **Resource Allocation**
   - Balance memory usage
   - Consider disk space
   - Monitor system resources
   - Handle cleanup gracefully

## Related Documentation
- [Monitoring Configuration](Monitoring.md#core-monitoring-settings) - Cache monitoring settings
- [Performance Guide](Performance.md) - Cache performance optimization
- [Troubleshooting Guide](Troubleshooting.md#performance-issues) - Cache-related issues
- [Configuration Guide](Config.md#cache-configuration) - Detailed cache configuration

## Components

### 1. CacheManager

The core component that orchestrates caching operations:

```python
from src.training.cache import CacheManager, CacheConfig

config = CacheConfig(
    memory_size="4GB",
    disk_size="100GB",
    policy="lru",
    eviction_threshold=0.9
)

cache_manager = CacheManager(config)
```

### 2. DatasetCache

Specialized cache for dataset management:

```python
from src.training.cache import DatasetCache

cache = DatasetCache()

# Cache dataset with optional preprocessing
cache.cache_dataset(dataset, preprocess_fn=None)

# Retrieve cached dataset
cached_data = cache.get_dataset("dataset_key")

# Stream through cache
for batch in cache.stream_dataset(dataset):
    process_batch(batch)
```

## Cache Policies

1. **LRU (Least Recently Used)**
   - Evicts least recently accessed items
   - Optimal for most use cases
   - Configurable cache size

2. **Size-based**
   - Evicts items based on memory pressure
   - Maintains memory usage below threshold
   - Automatic cleanup

## Storage Backends

### 1. Memory Cache
- Fast access times
- Limited by available RAM
- Configurable size limits
- Automatic garbage collection

### 2. Disk Cache
- Larger storage capacity
- Persistent across sessions
- Memory-mapped files
- Automatic cleanup

## Configuration

### Environment Variables

```bash
# Cache settings
CACHE_MEMORY_SIZE=4GB
CACHE_DISK_SIZE=100GB
CACHE_POLICY=lru
```

### Configuration File

```toml
[cache]
memory_size = "4GB"
disk_size = "100GB"
policy = "lru"
eviction_threshold = 0.9

[cache.memory]
max_items = 1000
cleanup_interval = 300

[cache.disk]
path = ".cache/training"
use_mmap = true
```

## Best Practices

### 1. Memory Management

- Set appropriate cache sizes
- Monitor memory usage
- Enable automatic cleanup
- Use streaming for large datasets

### 2. Performance Optimization

- Choose appropriate cache policy
- Enable memory mapping for large files
- Configure cleanup intervals
- Monitor cache hit rates

### 3. Data Integrity

- Implement proper error handling
- Verify cached data integrity
- Handle cache corruption
- Implement backup strategies

## Troubleshooting

### Common Issues

1. Memory Pressure
   - Reduce cache sizes
   - Enable more aggressive cleanup
   - Use disk cache for large datasets
   - Monitor memory usage

2. Slow Performance
   - Check cache hit rates
   - Optimize cache sizes
   - Enable memory mapping
   - Monitor I/O operations

3. Cache Corruption
   - Implement data validation
   - Use safe file operations
   - Enable backup strategies
   - Monitor disk health

## API Reference

### CacheManager

```python
class CacheManager:
    def __init__(self, config: CacheConfig):
        """Initialize cache manager with configuration."""
        
    def get(self, key: str) -> Any:
        """Retrieve item from cache."""
        
    def put(self, key: str, value: Any):
        """Store item in cache."""
        
    def remove(self, key: str):
        """Remove item from cache."""
        
    def clear(self):
        """Clear all cached items."""
```

### DatasetCache

```python
class DatasetCache:
    def __init__(self):
        """Initialize dataset cache."""
        
    def cache_dataset(self, dataset: Dataset, preprocess_fn: Optional[Callable] = None):
        """Cache dataset with optional preprocessing."""
        
    def get_dataset(self, key: str) -> Dataset:
        """Retrieve cached dataset."""
        
    def stream_dataset(self, dataset: Dataset) -> Iterator[Batch]:
        """Stream dataset through cache."""
```

## Future Extensions

1. Cache Features
   - New cache policies
   - Additional backends
   - Distributed caching
   - Cache compression

2. Performance
   - Cache prefetching
   - Smart eviction
   - Cache warming
   - Performance metrics

3. Integration
   - Custom backends
   - External caching systems
   - Monitoring tools
   - Analytics 