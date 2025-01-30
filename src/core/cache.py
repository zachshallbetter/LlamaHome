"""Cache implementations for model optimization."""

from typing import Any, Dict, Optional, Tuple, TypeVar

import torch

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
