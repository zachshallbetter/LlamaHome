# Cache System

## Table of Contents

- [Overview](#overview)
- [Cache Configuration](#cache-configuration)
- [Cache Types](#cache-types)
- [Cache Management](#cache-management)
- [Best Practices](#best-practices)

## Overview

LlamaHome implements a comprehensive caching system to optimize performance and resource usage. This document details the cache configuration, management, and best practices.

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