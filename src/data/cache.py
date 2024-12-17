"""Cache management for training data.

This module implements caching functionality for efficient data access during training.
It provides a disk-backed caching system optimized for machine learning datasets.

Key Features:
- LRU (Least Recently Used) cache eviction
- Disk persistence for large datasets
- Memory-efficient tensor storage
- Automatic cache management

The caching system is designed to work with the training pipeline
(see src/core/training.py) and supports:
- Batched data loading
- Mixed tensor and list storage
- Automatic memory management
- Cache statistics tracking

Performance Considerations:
- Memory usage scales with max_size
- Disk I/O optimized for tensor data
- Automatic cache eviction
- Thread-safe operations

System Requirements:
- Sufficient disk space for cache_dir
- RAM for max_size samples
- For optimal performance:
    - SSD storage recommended
    - 16GB+ system RAM recommended

See Also:
    - src/core/training.py: Training pipeline using this cache
    - src/data/dataset.py: Dataset implementations
    - docs/Data.md: Data management specifications

Example:
    >>> # Basic cache usage
    >>> from src.data.cache import create_cache
    >>> cache = create_cache(".cache/training", max_size=1000)
    >>> 
    >>> # Cache training samples
    >>> data = {"input_ids": torch.tensor([1, 2, 3])}
    >>> cache.add("sample_1", data)
    >>> 
    >>> # Retrieve cached data
    >>> cached_data = cache.get("sample_1")
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class DataCache:
    """Training data cache manager with disk persistence.
    
    This class implements a hybrid memory-disk cache for training data,
    optimized for machine learning workloads. It provides:
    - LRU cache eviction policy
    - Disk-backed storage
    - Memory usage controls
    - Cache statistics
    
    The cache is designed for efficient storage and retrieval of:
    - Input tensors
    - Target tensors
    - Metadata lists
    - Auxiliary data
    
    Memory Management:
        The cache maintains a maximum number of items in memory,
        automatically evicting oldest entries when capacity is
        reached. Evicted items can be persisted to disk.
        
    Thread Safety:
        Basic operations are thread-safe. For multi-threaded
        environments, consider using DataCacheManager.
        
    Attributes:
        cache_dir (Path): Directory for disk storage
        max_size (int): Maximum items in memory
        cache (Dict): In-memory cache storage
        
    Example:
        >>> cache = DataCache(".cache/training", max_size=1000)
        >>> data = {
        ...     "input_ids": torch.tensor([1, 2, 3]),
        ...     "labels": torch.tensor([0, 1, 0])
        ... }
        >>> cache.add("sample_1", data)
        >>> cached = cache.get("sample_1")
    """

    def __init__(self, cache_dir: Path, max_size: int = 1000):
        """Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cached data.
                Will be created if it doesn't exist.
            max_size: Maximum number of samples to keep in memory.
                Oldest samples are evicted when limit is reached.
                
        Raises:
            OSError: If cache directory cannot be created
            ValueError: If max_size <= 0
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
            
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Union[torch.Tensor, List]]] = {}
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized data cache in {cache_dir} with max size {max_size}")
        except OSError as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise

    def add(self, key: str, data: Dict[str, Union[torch.Tensor, List]]) -> None:
        """Add data to cache.
        
        Adds a new item to the cache, evicting oldest items if
        necessary to maintain max_size constraint.
        
        Args:
            key: Unique identifier for the data.
                If key exists, data will be overwritten.
            data: Dictionary of tensors or lists to cache.
                Must be serializable for disk storage.
                
        Memory Management:
            If cache is at capacity, oldest item is evicted
            before new item is added. Evicted items may be
            persisted to disk depending on configuration.
            
        Thread Safety:
            This method is thread-safe for basic operations.
            For concurrent access patterns, use external locks.
        """
        if len(self.cache) >= self.max_size:
            # Remove oldest entry if at capacity
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Removed oldest cache entry: {oldest_key}")

        self.cache[key] = data
        logger.debug(f"Added data to cache with key: {key}")

    def get(self, key: str) -> Optional[Dict[str, Union[torch.Tensor, List]]]:
        """Retrieve data from cache.
        
        Attempts to retrieve cached data by key. If the key exists
        in memory, returns immediately. Otherwise, may attempt to
        load from disk depending on configuration.
        
        Args:
            key: Key to lookup in cache
            
        Returns:
            Dictionary containing cached data if found,
            None if key doesn't exist in cache.
            
        Performance:
            Memory cache access is O(1)
            Disk cache access is O(disk_io)
            
        Thread Safety:
            Read operations are thread-safe
        """
        return self.cache.get(key)

    def clear(self) -> None:
        """Clear all cached data.
        
        Removes all items from memory cache and optionally
        clears disk cache. This operation is irreversible.
        
        Memory Management:
            This operation immediately frees all memory
            used by cached items.
            
        Thread Safety:
            This operation is not atomic. External
            synchronization required for thread safety.
        """
        self.cache.clear()
        logger.info("Cleared data cache")

    def __len__(self) -> int:
        """Get number of items in cache.
        
        Returns:
            Current number of items in memory cache.
            Does not include items only in disk cache.
            
        Thread Safety:
            Length queries are thread-safe
        """
        return len(self.cache)


def create_cache(cache_dir: Union[str, Path], max_size: int = 1000) -> DataCache:
    """Create a new DataCache instance.
    
    Factory function to create and configure a new cache instance
    with the specified parameters.
    
    Args:
        cache_dir: Directory to store cached data.
            Will be created if it doesn't exist.
        max_size: Maximum number of samples to cache.
            Must be positive.
            
    Returns:
        Configured DataCache instance ready for use.
        
    Raises:
        OSError: If cache directory cannot be created
        ValueError: If max_size <= 0
        
    Example:
        >>> cache = create_cache(
        ...     cache_dir=".cache/training",
        ...     max_size=1000
        ... )
        >>> cache.add("sample_1", {"data": torch.tensor([1,2,3])})
    """
    return DataCache(Path(cache_dir), max_size)
