"""Caching implementation with memory and disk tiers for training pipeline.

This module provides a two-tier caching system optimized for training data
and model artifacts. It implements both memory and disk caching with
automatic eviction policies.

Key Features:
- Memory-first caching with disk fallback
- LRU eviction policy
- Size-based constraints
- Automatic cleanup
"""
import sys
import asyncio
import hashlib
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, List, Callable, Iterator

import torch
from pydantic import BaseModel
from contextlib import contextmanager

from ..core.utils import LogManager, LogTemplates
from ..core.utils.io import (
    safe_load_json,
    safe_load_torch,
    safe_save_json,
    safe_save_torch,
)

from src.core.cache import CachePolicy, CachePersistence, MemoryManager

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

T = TypeVar("T")


class CacheConfig(BaseModel):
    """Cache configuration."""

    memory_size: int = 1000  # MB
    disk_size: int = 10000  # MB
    cleanup_interval: int = 3600  # seconds
    max_age_days: int = 7
    use_mmap: bool = True
    compression: bool = True
    async_writes: bool = True


class CacheItem(BaseModel):
    """Cache item metadata."""

    key: str
    size: int
    last_access: float
    path: Path | None = None


class Cache(ABC):
    """Abstract base class for cache implementations."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._data: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear cache."""
        pass

    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit."""
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)


class MemoryCache:
    """Memory-tier cache implementation."""

    def __init__(self, max_size: int):
        """Initialize memory cache.

        Args:
            max_size: Maximum cache size in MB
        """
        self.max_size = max_size * 1024 * 1024  # Convert to bytes
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._metadata: dict[str, CacheItem] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item if found
        """
        if key in self._data:
            value = self._data[key]
            # Move to end (most recently used)
            self._data.move_to_end(key)
            return value
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Item to cache
        """
        # Calculate size
        size = self._calculate_size(value)

        # Enforce size limit
        while self._current_size() + size > self.max_size and self._data:
            # Remove least recently used
            self._data.popitem(last=False)

        self._data[key] = value
        self._metadata[key] = CacheItem(
            key=key,
            size=size,
            last_access=time.time(),
        )

    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes.

        Args:
            value: Value to calculate size for

        Returns:
            Size in bytes
        """
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.nelement()
        return sys.getsizeof(value)

    def _current_size(self) -> int:
        """Get current cache size in bytes.

        Returns:
            Current size in bytes
        """
        return sum(item.size for item in self._metadata.values())


class DiskCache:
    """Disk-tier cache implementation."""

    def __init__(self, cache_dir: Path, max_size: int):
        """Initialize disk cache.

        Args:
            cache_dir: Cache directory
            max_size: Maximum cache size in MB
        """
        self.cache_dir = cache_dir
        self.max_size = max_size * 1024 * 1024  # Convert to bytes
        self._metadata: dict[str, CacheItem] = {}

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self._load_metadata()

    def get(self, key: str) -> Any | None:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item if found
        """
        if key in self._metadata:
            path = self._get_path(key)
            if path.exists():
                try:
                    if path.suffix in {".pt", ".pth"}:
                        return safe_load_torch(path)
                    return safe_load_json(path)
                except Exception:
                    # Remove invalid cache entry
                    self._remove(key)
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Item to cache
        """
        path = self._get_path(key)

        try:
            # Save value
            if isinstance(value, torch.Tensor):
                safe_save_torch(value, path)
            else:
                safe_save_json(value, path)

            # Update metadata
            self._metadata[key] = CacheItem(
                key=key,
                size=path.stat().st_size,
                last_access=time.time(),
                path=path,
            )

            # Enforce size limit
            self._enforce_size_limit()

            # Save metadata
            self._save_metadata()

        except Exception as e:
            # Clean up on error
            if path.exists():
                path.unlink()
            raise ValueError(f"Failed to cache item: {e}") from e

    def _get_path(self, key: str) -> Path:
        """Get cache file path for key.

        Args:
            key: Cache key

        Returns:
            Cache file path
        """
        # Use hash of key as filename
        filename = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / filename

    def _remove(self, key: str) -> None:
        """Remove item from cache.

        Args:
            key: Cache key
        """
        if key in self._metadata:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
            del self._metadata[key]

    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit by removing old items."""
        current_size = sum(item.size for item in self._metadata.values())

        # Sort by last access time
        items = sorted(
            self._metadata.items(),
            key=lambda x: x[1].last_access,
        )

        # Remove oldest items until under limit
        while current_size > self.max_size and items:
            key, item = items.pop(0)
            current_size -= item.size
            self._remove(key)

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_path = self.cache_dir / "metadata.json"
        if metadata_path.exists():
            try:
                data = safe_load_json(metadata_path)
                self._metadata = {k: CacheItem(**v) for k, v in data.items()}
            except Exception:
                # Start fresh if metadata is corrupt
                self._metadata = {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        metadata_path = self.cache_dir / "metadata.json"
        try:
            safe_save_json(
                {k: v.dict() for k, v in self._metadata.items()},
                metadata_path,
            )
        except Exception as e:
            raise ValueError(f"Failed to save cache metadata: {e}") from e


class CacheManager:
    """Two-tier cache manager implementation."""

    def __init__(self, config: CacheConfig):
        """Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.memory_cache = MemoryCache(config.memory_size)
        self.disk_cache = DiskCache(
            Path(".cache/training"),
            config.disk_size,
        )

    def get(self, key: str) -> Any | None:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item if found
        """
        # Try memory first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try disk
        value = self.disk_cache.get(key)
        if value is not None:
            # Cache in memory
            self.memory_cache.put(key, value)
            return value

        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Item to cache
        """
        # Cache in memory
        self.memory_cache.put(key, value)

        # Cache on disk
        self.disk_cache.put(key, value)

    def clear(self) -> None:
        """Clear all cache tiers."""
        self.memory_cache = MemoryCache(self.config.memory_size)
        self.disk_cache = DiskCache(
            Path(".cache/training"),
            self.config.disk_size,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for all tiers."""
        return {
            "memory": self.memory_cache._metadata,
            "disk": self.disk_cache._metadata,
        }


class CacheError(Exception):
    """Cache operation error."""

    pass


class DatasetCache:
    """Cache for dataset management."""

    def __init__(self, cache_dir: Path, config: Dict[str, Any]):
        """Initialize dataset cache.
        
        Args:
            cache_dir: Cache directory
            config: Cache configuration
        """
        self.persistence = CachePersistence(cache_dir, config)
        self.memory_manager = MemoryManager(config)

    def cache_dataset(
        self,
        name: str,
        dataset: List[Dict[str, Any]],
        preprocess_fn: Optional[Callable] = None
    ) -> None:
        """Cache dataset with optional preprocessing.
        
        Args:
            name: Dataset name
            dataset: Dataset to cache
            preprocess_fn: Optional preprocessing function
        """
        if preprocess_fn:
            dataset = [preprocess_fn(sample) for sample in dataset]
        self.persistence.save(name, dataset)

    def get_dataset(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Cached dataset if exists
        """
        return self.persistence.load(name)

    @contextmanager
    def stream_dataset(self, name: str, dataset: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Stream dataset through cache.
        
        Args:
            name: Dataset name
            dataset: Dataset to stream
            
        Yields:
            Dataset samples
        """
        batch_size = 32  # Could be configurable
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            if self.memory_manager.is_memory_available(100):  # 100MB buffer
                yield batch
            else:
                # Save to disk if memory constrained
                self.persistence.save(f"{name}_batch_{i}", batch)
                yield self.persistence.load(f"{name}_batch_{i}")


class TrainingCache:
    """Main training cache implementation."""

    def __init__(self, cache_dir: Path, config: Dict[str, Any]):
        """Initialize training cache.
        
        Args:
            cache_dir: Cache directory
            config: Cache configuration
        """
        self.config = config
        self.max_size = config["cache"].get("max_size", "4GB")
        self.policy = config["cache"].get("policy", "lru")
        self.persistence_enabled = config["cache"].get("persistence", True)
        self.compression_enabled = config["cache"].get("compression", True)
        
        self.cache_policy = CachePolicy(self.policy, config)
        self.persistence = CachePersistence(cache_dir, config)
        self.memory_manager = MemoryManager(config)

    def store(self, key: str, data: Any, category: str = "default") -> None:
        """Store data in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            category: Optional category
        """
        full_key = f"{category}/{key}"
        self.cache_policy.add_entry(full_key)
        self.persistence.save(full_key, data)

    def retrieve(self, key: str, category: str = "default") -> Optional[Any]:
        """Retrieve data from cache.
        
        Args:
            key: Cache key
            category: Optional category
            
        Returns:
            Cached data if exists
        """
        full_key = f"{category}/{key}"
        return self.persistence.load(full_key)

    def exists(self, key: str, category: str = "default") -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            category: Optional category
            
        Returns:
            Whether key exists
        """
        full_key = f"{category}/{key}"
        return self.persistence.exists(full_key)

    def get_category_size(self, category: str) -> int:
        """Get total size of category.
        
        Args:
            category: Cache category
            
        Returns:
            Size in bytes
        """
        # Implementation would track category sizes
        return 0

    def get_total_size(self) -> int:
        """Get total cache size.
        
        Returns:
            Size in bytes
        """
        # Implementation would track total size
        return 0

    def get_max_size(self) -> int:
        """Get maximum cache size.
        
        Returns:
            Size in bytes
        """
        # Parse max size string (e.g., "4GB")
        unit = self.max_size[-2:]
        size = float(self.max_size[:-2])
        if unit == "GB":
            return int(size * 1024 * 1024 * 1024)
        elif unit == "MB":
            return int(size * 1024 * 1024)
        return int(size)

    def cleanup(self) -> None:
        """Clean up cache to stay within size limits."""
        while self.get_total_size() > self.get_max_size():
            key = self.cache_policy.get_eviction_candidate()
            if key:
                self.persistence.remove(key)
                self.cache_policy.remove_entry(key)
