"""Cache implementations for model optimization."""

from typing import Any, Dict, Optional, Tuple, TypeVar, Union
from pathlib import Path
import torch
import psutil

from src.core.utils.log_manager import LogManager, LogTemplates


logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

T = TypeVar("T")  # For generic type hints


class BaseCache:
    """Base class for all cache implementations."""

    def __init__(self, max_length: int):
        """Initialize base cache.

        Args:
            max_length: Maximum cache length
        """
        self.max_length = max_length
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get cached states for layer.

        Args:
            layer_idx: Layer index

        Returns:
            Cached states if found
        """
        if layer_idx not in self.cache:
            return None
        return (self.cache[layer_idx]["keys"], self.cache[layer_idx]["values"])

    def update(
        self,
        layer_idx: int,
        states: Tuple[torch.Tensor, ...],
        position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> None:
        """Update cache with new states.

        Args:
            layer_idx: Layer index to update
            states: Tuple of (key_states, value_states) to cache
            position: Optional position tensor for specific updates
            **kwargs: Additional arguments for future extensions
        """
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {
                "keys": states[0].detach(),
                "values": states[1].detach(),
            }
        else:
            if position is not None:
                # Update specific positions
                self.cache[layer_idx]["keys"][:, :, position] = states[0]
                self.cache[layer_idx]["values"][:, :, position] = states[1]
            else:
                # Append new states
                self.cache[layer_idx]["keys"] = torch.cat(
                    [self.cache[layer_idx]["keys"], states[0]], dim=-2
                )
                self.cache[layer_idx]["values"] = torch.cat(
                    [self.cache[layer_idx]["values"], states[1]], dim=-2
                )

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Clear cache for layer or all layers.

        Args:
            layer_idx: Optional layer index to clear
        """
        if layer_idx is not None:
            if layer_idx in self.cache:
                del self.cache[layer_idx]
        else:
            self.cache.clear()

    def get_seq_length(self, layer_idx: int) -> int:
        """Get sequence length for layer.

        Args:
            layer_idx: Layer index

        Returns:
            Current sequence length
        """
        if layer_idx not in self.cache:
            return 0
        return self.cache[layer_idx]["keys"].size(-2)


class DynamicCache(BaseCache):
    """Dynamic cache implementation."""

    def __init__(self, initial_length: int = 1024):
        """Initialize dynamic cache.

        Args:
            initial_length: Initial cache length
        """
        super().__init__(initial_length)

    def update(
        self,
        layer_idx: int,
        states: Tuple[torch.Tensor, ...],
        position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> None:
        """Update cache with dynamic resizing.

        Args:
            layer_idx: Layer index to update
            states: Tuple of (key_states, value_states) to cache
            position: Optional position tensor for specific updates
            **kwargs: Additional arguments for future extensions
        """
        super().update(layer_idx, states, position, **kwargs)

        # Grow cache if needed
        current_length = self.get_seq_length(layer_idx)
        if current_length > self.max_length:
            self.max_length = min(current_length * 2, 16384)


class StaticCache(BaseCache):
    """Static cache with fixed length."""

    def __init__(self, length: int):
        """Initialize static cache.

        Args:
            length: Fixed maximum cache length
        """
        super().__init__(length)

    def update(
        self,
        layer_idx: int,
        states: Tuple[torch.Tensor, ...],
        position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> None:
        """Update cache with fixed length enforcement.

        Args:
            layer_idx: Layer index to update
            states: Tuple of (key_states, value_states) to cache
            position: Optional position tensor for specific updates
            **kwargs: Additional arguments for future extensions
        """
        super().update(layer_idx, states, position, **kwargs)

        # Ensure we don't exceed fixed length
        current_length = self.get_seq_length(layer_idx)
        if current_length > self.max_length:
            self.cache[layer_idx]["keys"] = self.cache[layer_idx]["keys"][
                :, :, -self.max_length :
            ]
            self.cache[layer_idx]["values"] = self.cache[layer_idx]["values"][
                :, :, -self.max_length :
            ]


class CachePolicy:
    """Manages cache eviction policies."""

    def __init__(self, policy_type: str = "fifo", config: Optional[Dict[str, Any]] = None):
        """Initialize cache policy.
        
        Args:
            policy_type: Type of cache policy ("fifo", "lru", "lfu")
            config: Optional configuration
        """
        self.policy_type = policy_type
        self.config = config or {}
        self.entries = []

    def add_entry(self, key: str) -> None:
        """Add entry to cache.
        
        Args:
            key: Cache key
        """
        if self.policy_type == "fifo":
            self.entries.append(key)
        elif self.policy_type == "lru":
            if key in self.entries:
                self.entries.remove(key)
            self.entries.append(key)

    def get_eviction_candidate(self) -> Optional[str]:
        """Get next entry to evict.
        
        Returns:
            Key to evict or None
        """
        if not self.entries:
            return None
        return self.entries[0]

    def remove_entry(self, key: str) -> None:
        """Remove entry from cache.
        
        Args:
            key: Cache key
        """
        if key in self.entries:
            self.entries.remove(key)


class CachePersistence:
    """Manages cache persistence to disk."""

    def __init__(self, cache_dir: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """Initialize cache persistence.
        
        Args:
            cache_dir: Cache directory
            config: Optional configuration
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def save(self, key: str, data: Any) -> None:
        """Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        cache_path = self.cache_dir / f"{key}.pt"
        torch.save(data, cache_path)

    def load(self, key: str) -> Optional[Any]:
        """Load data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if exists
        """
        cache_path = self.cache_dir / f"{key}.pt"
        if cache_path.exists():
            return torch.load(cache_path)
        return None

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            Whether key exists
        """
        cache_path = self.cache_dir / f"{key}.pt"
        return cache_path.exists()

    def remove(self, key: str) -> None:
        """Remove key from cache.
        
        Args:
            key: Cache key
        """
        cache_path = self.cache_dir / f"{key}.pt"
        if cache_path.exists():
            cache_path.unlink()


class MemoryManager:
    """Manages memory usage and limits."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory manager.
        
        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.memory_limit = self.config.get("memory_limit", None)

    def check_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "ram_used": psutil.Process().memory_info().rss / (1024 * 1024),  # MB
            "ram_percent": psutil.Process().memory_percent(),
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_allocated": torch.cuda.memory_allocated() / (1024 * 1024),  # MB
                "gpu_cached": torch.cuda.memory_reserved() / (1024 * 1024),  # MB
            })
            
        return stats

    def is_memory_available(self, required_mb: float) -> bool:
        """Check if memory is available.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            Whether memory is available
        """
        if self.memory_limit:
            current_usage = self.check_memory_usage()["ram_used"]
            return current_usage + required_mb <= self.memory_limit
        return True
