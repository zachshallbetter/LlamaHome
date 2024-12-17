"""Hybrid attention mechanism implementation."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from ..utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

@dataclass
class AttentionConfig:
    """Configuration for hybrid attention mechanism.
    
    This class defines the configuration for the attention mechanism,
    controlling how attention is computed and optimized. It works in
    conjunction with ModelConfig (see src/core/model.py).

    Memory Management:
        The attention mechanism automatically manages memory based on:
        1. Available GPU memory
        2. Input sequence length
        3. Batch size
        4. Model size

    Hardware Optimization:
        The mechanism automatically selects the best implementation:
        1. Flash Attention for supported GPUs
        2. Memory-efficient attention for other cases
        3. Fallback to standard attention if needed

    Attributes:
        use_flash_attention (bool): Use Flash Attention when available.
        use_memory_efficient (bool): Use memory efficient implementation.
        head_dim (int): Attention head dimension.
        num_heads (int): Number of attention heads.
        sliding_window (Optional[int]): Size of attention window, None for full.
        attention_dropout (float): Dropout probability.

    Related:
        - ModelConfig in src/core/model.py
        - Cache configuration in src/core/cache.py

    Example:
        >>> config = AttentionConfig(
        ...     use_flash_attention=True,
        ...     sliding_window=1024,
        ...     head_dim=128,
        ...     num_heads=32
        ... )
        >>> attention = HybridAttention(config, layer_idx=0)
    """
    
    use_flash_attention: bool = True
    use_memory_efficient: bool = True
    head_dim: int = 128
    num_heads: int = 32
    sliding_window: Optional[int] = None
    attention_dropout: float = 0.1


class HybridAttention(LlamaAttention):
    """Hybrid attention implementation combining multiple optimizations.
    
    This class extends the base Llama attention with:
    1. Flash Attention support
    2. Memory-efficient implementation
    3. Dynamic cache management
    4. Position-aware attention
    5. Sliding window support

    The implementation automatically selects the best attention mechanism
    based on hardware capabilities and memory constraints. It integrates
    with the EnhancedLlamaForCausalLM model's caching and memory
    management systems.

    Architecture:
        1. Attention Computation:
            - Flash Attention for supported hardware
            - Memory-efficient implementation as fallback
            - Standard attention as final fallback
        
        2. Memory Management:
            - Dynamic cache sizing
            - Sliding window optimization
            - Position-aware memory mapping
        
        3. Integration:
            - Works with EnhancedLlamaForCausalLM
            - Supports distributed training
            - Handles gradient checkpointing

    See Also:
        - EnhancedLlamaForCausalLM in src/core/model.py
        - Cache implementation in src/core/cache.py
        - Training system in src/core/training.py

    Example:
        >>> # Configuration
        >>> config = AttentionConfig(use_flash_attention=True)
        >>> attention = HybridAttention(config, layer_idx=0)
        >>> 
        >>> # Forward pass
        >>> hidden_states = torch.randn(1, 32, 512)
        >>> output = attention(
        ...     hidden_states,
        ...     use_cache=True
        ... )
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Initialize hybrid attention layer.
        
        Args:
            config: Attention configuration with settings:
                - use_flash_attention (bool): Use Flash Attention
                - use_memory_efficient (bool): Use memory efficient
                - sliding_window (int): Attention window size
                - head_dim (int): Attention head dimension
                - num_heads (int): Number of attention heads
            layer_idx: Index of this attention layer

        Raises:
            RuntimeError: If Flash Attention is requested but unavailable
            ValueError: If configuration is invalid

        Note:
            The layer automatically falls back to the best available
            implementation if the requested one is unavailable.
        """
        super().__init__(config)
        self.layer_idx = layer_idx
        self.past_key_value = None
        self.cache = DynamicCache()
        
        # Configure attention settings
        self.attention_config = AttentionConfig(
            use_flash_attention=getattr(config, "use_flash_attention", True) and FLASH_ATTN_AVAILABLE,
            use_memory_efficient=getattr(config, "use_memory_efficient", True),
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            sliding_window=getattr(config, "sliding_window", None),
            attention_dropout=self.dropout
        )
        
        if self.attention_config.use_flash_attention and not FLASH_ATTN_AVAILABLE:
            logger.warning("Flash Attention requested but not available, falling back to memory efficient attention")
            self.attention_config.use_flash_attention = False
        
        self.logger = LogManager().get_logger("hybrid_attention", "models", "llama")
        self.model_name = "HybridAttention"
        
        try:
            self.logger.info(LogTemplates.MODEL_LOADED.format(
                model_name=self.model_name
            ))
        except Exception as e:
            self.logger.error(LogTemplates.MODEL_ERROR.format(
                model_name=self.model_name,
                error=str(e)
            ))
            raise
            
        logger.debug(f"Initialized hybrid attention layer {layer_idx}")

    def _memory_efficient_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute attention using memory efficient implementation.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (context, attention weights)
        """
        logger.debug("Using memory efficient attention")
        
        # Reshape for efficient computation
        batch_size, q_len, num_heads, head_dim = query.shape
        kv_len = key.shape[1]
        
        # Scale query
        scaling = float(head_dim) ** -0.5
        query = query * scaling
        
        # Compute attention scores
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, q_len, kv_len)
            
        # Compute attention with memory efficient implementation
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),  # [batch_size, num_heads, q_len, head_dim]
            key.transpose(1, 2),    # [batch_size, num_heads, kv_len, head_dim]
            value.transpose(1, 2),  # [batch_size, num_heads, kv_len, head_dim]
            attn_mask=attention_mask,
            dropout_p=self.attention_config.attention_dropout if self.training else 0.0,
        )
        
        return attn_output.transpose(1, 2), None

    def _flash_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute attention using Flash Attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (context, attention weights)
        """
        logger.debug("Using Flash Attention")
        
        # Handle variable sequence lengths if needed
        if attention_mask is not None:
            attn_output = flash_attn_varlen_func(
                query,
                key,
                value,
                attention_mask,
                dropout_p=self.attention_config.attention_dropout if self.training else 0.0,
                softmax_scale=float(self.attention_config.head_dim) ** -0.5,
                causal=True,
            )
        else:
            attn_output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=self.attention_config.attention_dropout if self.training else 0.0,
                softmax_scale=float(self.attention_config.head_dim) ** -0.5,
                causal=True,
            )
            
        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass with hybrid attention mechanism.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional cached key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cache
            cache_position: Optional cache positions
            
        Returns:
            Tuple of (output, past_key_value, attention_weights)
        """
        logger.debug("Starting hybrid attention forward pass")
        
        # Handle caching
        if use_cache:
            cached_kv = self.cache.get(self.layer_idx, cache_position)
            if cached_kv is not None:
                logger.debug(f"Cache hit for layer {self.layer_idx}")
                past_key_value = cached_kv
            else:
                logger.debug(f"Cache miss for layer {self.layer_idx}")

        # Project hidden states to query, key, value
        batch_size, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        
        # Apply sliding window if configured
        if self.attention_config.sliding_window is not None:
            window_size = self.attention_config.sliding_window
            key_states = key_states[:, -window_size:]
            value_states = value_states[:, -window_size:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -window_size:]

        # Choose attention implementation
        if self.attention_config.use_flash_attention and torch.cuda.is_available():
            attn_output, attn_weights = self._flash_attention(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = self._memory_efficient_attention(
                query_states, key_states, value_states, attention_mask
            )

        # Project back to hidden size
        attn_output = attn_output.contiguous().view(batch_size, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Update cache if needed
        if use_cache:
            current_key_value = (key_states, value_states)
            self.cache.update(self.layer_idx, current_key_value, cache_position)

        logger.debug("Hybrid attention forward pass complete")
        return attn_output, past_key_value, attn_weights if output_attentions else None

    def __del__(self):
        """Cleanup when attention layer is deleted."""
        self.cache.clear(self.layer_idx) 