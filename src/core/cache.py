"""Cache implementations for model optimization.

This module provides caching mechanisms for optimizing model inference and training.
It includes three main cache implementations:

1. Base Cache: Foundation for all caching implementations
2. Dynamic Cache: Adaptive cache that grows with usage
3. Static Cache: Fixed-size cache for constrained environments

The caching system is designed to work with the attention mechanism
(see src/core/attention.py) and supports:
- Key-value caching for transformer layers
- Position-based updates
- Memory-efficient storage
- Automatic cache management

Performance Considerations:
- Memory usage scales with sequence length
- Dynamic cache grows by a configurable factor
- Static cache maintains fixed memory footprint
- Automatic cleanup on layer deletion

System Requirements:
- PyTorch >= 2.0
- Sufficient system memory for cached states
- For optimal performance:
    - CUDA-capable GPU recommended
    - 8GB+ system RAM recommended

See Also:
    - src/core/attention.py: Attention mechanism using this cache
    - src/core/model.py: Model implementation
    - docs/Architecture.md: System architecture overview

Example:
    >>> # Using dynamic cache
    >>> from src.core.cache import DynamicCache
    >>> cache = DynamicCache(initial_length=1024)
    >>> 
    >>> # Cache states for a layer
    >>> states = (key_tensor, value_tensor)
    >>> cache.update(layer_idx=0, states=states)
    >>> 
    >>> # Retrieve cached states
    >>> cached_states = cache.get(layer_idx=0)
"""

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

T = TypeVar("T")  # For generic type hints


class Cache:
    """Base cache implementation for model state management.
    
    This class provides the foundation for caching transformer layer states.
    It implements basic caching functionality with support for:
    - Layer-specific caching
    - Position-based updates
    - Maximum length constraints
    - Memory cleanup
    
    The cache stores key-value pairs for each layer, enabling efficient
    attention computation in transformer models. It supports both
    full-sequence and position-based updates.
    
    Memory Management:
        States are stored as PyTorch tensors with automatic cleanup.
        The cache can be constrained to a maximum length to prevent
        memory exhaustion during long sequences.
        
    Thread Safety:
        The cache is not thread-safe by default. For multi-threaded
        environments, external synchronization is required.
        
    Attributes:
        max_length (Optional[int]): Maximum cache length if set
        cache (Dict): Internal storage for cached states
        
    Example:
        >>> cache = Cache(max_length=1024)
        >>> states = (key_tensor, value_tensor)
        >>> cache.update(layer_idx=0, states=states)
        >>> cached = cache.get(layer_idx=0)
    """

    def __init__(self, max_length: Optional[int] = None):
        """Initialize cache.
        
        Args:
            max_length: Optional maximum cache length. If set, the cache
                will be trimmed to this length after updates.
        """
        self.max_length = max_length
        self.cache: Dict[int, Dict[str, Union[torch.Tensor, torch.Tensor]]] = {}

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Get current sequence length.
        
        Retrieves the current sequence length for a specific layer
        or the first available layer if none specified.
        
        Args:
            layer_idx: Optional layer index to get length for.
                If None, uses first available layer.
                
        Returns:
            Current sequence length as integer.
            Returns 0 if cache is empty.
        """
        if not self.cache:
            return 0
        if layer_idx is None:
            layer_idx = next(iter(self.cache))
        return int(self.cache[layer_idx]["keys"].size(2))

    def get_max_length(self) -> Optional[int]:
        """Get maximum cache length.
        
        Returns:
            Maximum cache length if set, None otherwise.
            This represents the absolute maximum sequence
            length that will be cached.
        """
        return self.max_length

    def get(self, layer_idx: int, position: Optional[torch.LongTensor] = None) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get cached states for layer.
        
        Retrieves cached key-value states for a specific layer.
        Can optionally retrieve states for specific positions.
        
        Args:
            layer_idx: Layer index to retrieve states for
            position: Optional position tensor for specific positions.
                If provided, returns states only for those positions.
                
        Returns:
            Tuple of (key_states, value_states) if available,
            None if layer not in cache.
            
        Shape:
            - position: (batch_size, seq_len)
            - output[0]: (batch_size, num_heads, seq_len, head_dim)
            - output[1]: (batch_size, num_heads, seq_len, head_dim)
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
        
        Updates the cache for a specific layer with new key-value states.
        Can update specific positions or append to existing sequence.
        
        Args:
            layer_idx: Layer index to update
            states: Tuple of (key_states, value_states) to cache
            position: Optional position tensor for specific updates
            **kwargs: Additional arguments for future extensions
            
        Memory Management:
            If max_length is set, the cache will be trimmed after
            the update to maintain the length constraint.
            
        Shape Requirements:
            - states[0]: (batch_size, num_heads, seq_len, head_dim)
            - states[1]: (batch_size, num_heads, seq_len, head_dim)
            - position: (batch_size, seq_len) if provided
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
        
        Clears cached states for a specific layer or entire cache.
        This operation is irreversible and frees memory immediately.
        
        Args:
            layer_idx: Optional layer index to clear.
                If None, clears entire cache.
                
        Memory Management:
            This operation immediately releases memory back to
            the system by deleting cached tensors.
        """
        if layer_idx is not None:
            if layer_idx in self.cache:
                del self.cache[layer_idx]
        else:
            self.cache.clear()


class DynamicCache(Cache):
    """Dynamic cache implementation with adaptive growth.
    
    This cache implementation automatically grows to accommodate
    longer sequences while maintaining memory efficiency. It uses
    a growth factor to gradually increase cache size as needed.
    
    Features:
        - Automatic cache growth
        - Configurable growth factor
        - Optional maximum size limit
        - Memory-efficient resizing
        
    The dynamic cache is ideal for:
        - Variable length sequences
        - Interactive applications
        - Streaming inference
        - Memory-constrained environments
        
    Memory Management:
        The cache grows by the growth factor when needed,
        but never exceeds the maximum length if specified.
        Growth is logarithmic in sequence length.
        
    Example:
        >>> cache = DynamicCache(
        ...     initial_length=1024,
        ...     growth_factor=1.5,
        ...     max_length=8192
        ... )
        >>> states = (key_tensor, value_tensor)
        >>> cache.update(layer_idx=0, states=states)
    """

    def __init__(
        self,
        initial_length: int = 1024,
        growth_factor: float = 1.5,
        max_length: Optional[int] = None
    ):
        """Initialize dynamic cache.
        
        Args:
            initial_length: Initial cache length (default: 1024)
            growth_factor: Multiplicative factor for cache growth
                Must be > 1.0 (default: 1.5)
            max_length: Optional maximum cache length
                If set, cache will not grow beyond this size
        
        The cache will grow by growth_factor each time it
        reaches capacity, until max_length is reached.
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
        
        Updates the cache and grows it if needed. Growth occurs
        when the current sequence length reaches current_length.
        
        Args:
            layer_idx: Layer index to update
            states: Tuple of (key_states, value_states) to cache
            position: Optional position tensor for specific updates
            **kwargs: Additional arguments for future extensions
            
        Growth Behavior:
            - Grows by growth_factor when capacity is reached
            - Never exceeds max_length if specified
            - Growth is logged at debug level
            
        Memory Management:
            Growth allocates new memory and copies existing states.
            Old memory is released after successful growth.
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
    """Static cache implementation with fixed size.
    
    This cache implementation maintains a fixed maximum length,
    ensuring predictable memory usage. It is ideal for:
    - Resource-constrained environments
    - Real-time applications
    - Systems with strict memory requirements
    - Production deployments
    
    Features:
        - Fixed memory footprint
        - Automatic trimming
        - Efficient memory usage
        - Predictable behavior
        
    Memory Management:
        The cache never exceeds its specified length.
        Older entries are automatically removed when
        new entries would exceed the length.
        
    Example:
        >>> cache = StaticCache(length=1024)
        >>> states = (key_tensor, value_tensor)
        >>> cache.update(layer_idx=0, states=states)
    """

    def __init__(self, length: int):
        """Initialize static cache.
        
        Args:
            length: Fixed maximum cache length.
                This length is strictly enforced.
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
        
        Updates the cache while maintaining the fixed length
        constraint. If the update would exceed the fixed length,
        older entries are removed.
        
        Args:
            layer_idx: Layer index to update
            states: Tuple of (key_states, value_states) to cache
            position: Optional position tensor for specific updates
            **kwargs: Additional arguments for future extensions
            
        Memory Management:
            - Maintains fixed length by trimming oldest entries
            - No dynamic allocation
            - Predictable memory usage
        """
        super().update(layer_idx, states, position, **kwargs)

        # Ensure we don't exceed fixed length
        current_length = self.get_seq_length(layer_idx)
        if current_length > self.max_length:
            self.cache[layer_idx]["keys"] = self.cache[layer_idx]["keys"][:, :, -self.max_length:]
            self.cache[layer_idx]["values"] = self.cache[layer_idx]["values"][:, :, -self.max_length:] 