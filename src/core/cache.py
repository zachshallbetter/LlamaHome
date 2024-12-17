"""Cache implementations for model optimization."""

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

T = TypeVar("T")  # For generic type hints


class Cache:
    """Base cache class."""

    def __init__(self, max_length: Optional[int] = None):
        """Initialize cache.
        
        Args:
            max_length: Optional maximum cache length
        """
        self.max_length = max_length
        self.cache: Dict[int, Dict[str, Union[torch.Tensor, torch.Tensor]]] = {}

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Get current sequence length.
        
        Args:
            layer_idx: Optional layer index to get length for
            
        Returns:
            Current sequence length
        """
        if not self.cache:
            return 0
        if layer_idx is None:
            layer_idx = next(iter(self.cache))
        return int(self.cache[layer_idx]["keys"].size(2))

    def get_max_length(self) -> Optional[int]:
        """Get maximum cache length.
        
        Returns:
            Maximum cache length if set
        """
        return self.max_length

    def get(self, layer_idx: int, position: Optional[torch.LongTensor] = None) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get cached states for layer.
        
        Args:
            layer_idx: Layer index
            position: Optional position tensor
            
        Returns:
            Cached states if available
        """
        if layer_idx not in self.cache:
            return None
            
        if position is None:
            return (self.cache[layer_idx]["keys"], self.cache[layer_idx]["values"])
            
        return (
            self.cache[layer_idx]["keys"][:, :, position],
            self.cache[layer_idx]["values"][:, :, position]
        )

    def update(
        self,
        layer_idx: int,
        states: Tuple[torch.Tensor, ...],
        position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> None:
        """Update cache with new states.
        
        Args:
            layer_idx: Layer index
            states: New states to cache
            position: Optional position tensor
            **kwargs: Additional arguments
        """
        key_states, value_states = states

        if layer_idx not in self.cache:
            self.cache[layer_idx] = {
                "keys": key_states,
                "values": value_states
            }
            return

        if position is not None:
            self.cache[layer_idx]["keys"][:, :, position] = key_states
            self.cache[layer_idx]["values"][:, :, position] = value_states
        else:
            self.cache[layer_idx]["keys"] = torch.cat(
                [self.cache[layer_idx]["keys"], key_states], dim=2
            )
            self.cache[layer_idx]["values"] = torch.cat(
                [self.cache[layer_idx]["values"], value_states], dim=2
            )

        # Trim cache if max length exceeded
        if self.max_length is not None:
            current_length = self.get_seq_length(layer_idx)
            if current_length > self.max_length:
                self.cache[layer_idx]["keys"] = self.cache[layer_idx]["keys"][:, :, -self.max_length:]
                self.cache[layer_idx]["values"] = self.cache[layer_idx]["values"][:, :, -self.max_length:]

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Clear cache.
        
        Args:
            layer_idx: Optional layer index to clear
        """
        if layer_idx is not None:
            if layer_idx in self.cache:
                del self.cache[layer_idx]
        else:
            self.cache.clear()


class DynamicCache(Cache):
    """Dynamic cache with adaptive length."""

    def __init__(
        self,
        initial_length: int = 1024,
        growth_factor: float = 1.5,
        max_length: Optional[int] = None
    ):
        """Initialize dynamic cache.
        
        Args:
            initial_length: Initial cache length
            growth_factor: Cache growth factor
            max_length: Optional maximum cache length
        """
        super().__init__(max_length)
        self.current_length = initial_length
        self.growth_factor = growth_factor

    def update(
        self,
        layer_idx: int,
        states: Tuple[torch.Tensor, ...],
        position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> None:
        """Update cache with dynamic resizing.
        
        Args:
            layer_idx: Layer index
            states: New states to cache
            position: Optional position tensor
            **kwargs: Additional arguments
        """
        super().update(layer_idx, states, position, **kwargs)

        # Check if we need to grow the cache
        current_length = self.get_seq_length(layer_idx)
        if current_length >= self.current_length:
            new_length = min(
                int(self.current_length * self.growth_factor),
                self.max_length or float("inf")
            )
            if new_length > self.current_length:
                self.current_length = new_length
                logger.debug(f"Growing cache to length {self.current_length}")


class StaticCache(Cache):
    """Static cache with fixed length."""

    def __init__(self, length: int):
        """Initialize static cache.
        
        Args:
            length: Fixed cache length
        """
        super().__init__(length)

    def update(
        self,
        layer_idx: int,
        states: Tuple[torch.Tensor, ...],
        position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> None:
        """Update cache with fixed length.
        
        Args:
            layer_idx: Layer index
            states: New states to cache
            position: Optional position tensor
            **kwargs: Additional arguments
        """
        super().update(layer_idx, states, position, **kwargs)

        # Ensure we don't exceed fixed length
        current_length = self.get_seq_length(layer_idx)
        if current_length > self.max_length:
            self.cache[layer_idx]["keys"] = self.cache[layer_idx]["keys"][:, :, -self.max_length:]
            self.cache[layer_idx]["values"] = self.cache[layer_idx]["values"][:, :, -self.max_length:] 