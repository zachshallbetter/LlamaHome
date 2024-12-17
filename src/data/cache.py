"""Cache management for training data.

This module implements caching functionality for efficient data access during training,
following the specifications in docs/Data.md.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

class DataCache:
    """Manages caching of training data samples."""

    def __init__(self, cache_dir: Path, max_size: int = 1000):
        """Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cached data
            max_size: Maximum number of samples to cache
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Union[torch.Tensor, List]]] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized data cache in {cache_dir} with max size {max_size}")

    def add(self, key: str, data: Dict[str, Union[torch.Tensor, List]]) -> None:
        """Add data to cache.
        
        Args:
            key: Unique identifier for the data
            data: Dictionary containing tensors or lists to cache
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
        
        Args:
            key: Key to lookup in cache
            
        Returns:
            Cached data if found, None otherwise
        """
        return self.cache.get(key)

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cleared data cache")

    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self.cache)


def create_cache(cache_dir: Union[str, Path], max_size: int = 1000) -> DataCache:
    """Create a new DataCache instance.
    
    Args:
        cache_dir: Directory to store cached data
        max_size: Maximum number of samples to cache
        
    Returns:
        Configured DataCache instance
    """
    return DataCache(Path(cache_dir), max_size)
