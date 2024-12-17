"""Cache management utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils import LogManager, LogTemplates


class CacheManager:
    """Manages caching of data and model artifacts."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager.
        
        Args:
            cache_dir: Optional custom cache directory
        """
        self.logger = LogManager(LogTemplates.CACHE).get_logger(__name__)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "llamahome"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Error reading cache for {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w") as f:
                json.dump(value, f)
            return True
        except Exception as e:
            self.logger.warning(f"Error writing cache for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                cache_path.unlink()
            return True
        except Exception as e:
            self.logger.warning(f"Error deleting cache for {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cached items.
        
        Returns:
            True if successful
        """
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            return True
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {e}")
            return False

    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{key}.json"