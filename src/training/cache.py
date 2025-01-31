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

import asyncio
import json
import mmap
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

"""
Caching implementation with memory and disk tiers for training pipeline.
"""


@dataclass
class CacheConfig:
    """Cache configuration."""

    cache_dir: Union[str, Path] = Path(".cache")
    max_size: int = 1000  # MB
    cleanup_interval: int = 3600  # seconds


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


class MemoryCache(Cache):
    """In-memory LRU cache."""

    def __init__(self, max_size: int = CacheConfig.memory_size):
        super().__init__(max_size)
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self._lock:
            if key in self._data:
                self._hits += 1
                value = self._data.pop(key)
                self._data[key] = value
                return value
            self._misses += 1
            return None

    async def put(self, key: str, value: Any) -> None:
        """Put value in memory cache."""
        async with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            self._enforce_size_limit()

    async def clear(self) -> None:
        """Clear memory cache."""
        async with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._data),
            "max_size": self.max_size,
        }


class DiskCache(Cache):
    """Disk-based persistent cache."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size: int = CacheConfig.disk_size,
        config: Optional[CacheConfig] = None,
    ):
        super().__init__(max_size)
        self.config = config or CacheConfig()
        self.cache_dir = Path(cache_dir)
        self._setup_cache_dir()
        self._load_metadata()

    def _setup_cache_dir(self) -> None:
        """Set up cache directory structure."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.meta_file = self.cache_dir / "metadata.json"

    def _load_metadata(self) -> None:
        """Load cache metadata."""
        if self.meta_file.exists():
            with open(self.meta_file, "r") as f:
                metadata = json.load(f)
                self._data = OrderedDict(metadata)
        else:
            self._data = OrderedDict()

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self.meta_file, "w") as f:
            json.dump(dict(self._data), f)

    def _get_path(self, key: str) -> Path:
        """Get path for cached item."""
        return self.data_dir / f"{key}.pkl"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        async with self._lock:
            if key not in self._data:
                return None

            path = self._get_path(key)
            if not path.exists():
                return None

            try:
                if self.config.use_mmap:
                    return self._load_mmap(path)
                return self._load_pickle(path)
            except Exception:
                return None

    async def put(self, key: str, value: Any) -> None:
        """Put value in disk cache."""
        async with self._lock:
            path = self._get_path(key)

            if self.config.async_writes:
                await self._async_save(path, value)
            else:
                self._sync_save(path, value)

            self._data[key] = {"path": str(path), "timestamp": time.time()}
            self._enforce_size_limit()
            self._save_metadata()

    async def clear(self) -> None:
        """Clear disk cache."""
        async with self._lock:
            for path in self.data_dir.glob("*.pkl"):
                path.unlink()
            self._data.clear()
            self._save_metadata()

    def _load_pickle(self, path: Path) -> Any:
        """Load pickled data."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_mmap(self, path: Path) -> Any:
        """Load memory-mapped data."""
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            return pickle.loads(mm)

    def _sync_save(self, path: Path, value: Any) -> None:
        """Save data synchronously."""
        with open(path, "wb") as f:
            pickle.dump(value, f)

    async def _async_save(self, path: Path, value: Any) -> None:
        """Save data asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_save, path, value)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(path.stat().st_size for path in self.data_dir.glob("*.pkl"))
        return {
            "items": len(self._data),
            "max_items": self.max_size,
            "total_size_bytes": total_size,
            "compression": self.config.compression,
            "mmap_enabled": self.config.use_mmap,
        }


class CacheManager:
    """Cache management with multiple tiers."""

    def __init__(
        self, cache_dir: Union[str, Path], config: Optional[CacheConfig] = None
    ) -> None:
        self.config = config or CacheConfig()
        self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.memory_cache = MemoryCache(self.config.memory_size)
        self.disk_cache = DiskCache(self.cache_dir, self.config.disk_size, self.config)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        # Try memory cache first
        if value := await self.memory_cache.get(key):
            return value

        # Try disk cache
        if value := await self.disk_cache.get(key):
            # Promote to memory cache
            await self.memory_cache.put(key, value)
            return value

        return None

    async def put(self, key: str, value: Any) -> None:
        """Put value in cache hierarchy."""
        # Save to both tiers
        await asyncio.gather(
            self.memory_cache.put(key, value), self.disk_cache.put(key, value)
        )

    async def clear(self) -> None:
        """Clear all cache tiers."""
        await asyncio.gather(self.memory_cache.clear(), self.disk_cache.clear())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for all tiers."""
        return {
            "memory": self.memory_cache.get_stats(),
            "disk": self.disk_cache.get_stats(),
        }


class CacheError(Exception):
    """Cache operation error."""

    pass
