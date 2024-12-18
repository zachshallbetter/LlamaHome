"""Hybrid attention mechanism implementation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaAttention



try:

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from ..utils import LogManager, LogTemplates
from ..cache import DynamicCache



logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass


class AttentionConfig:
    """Configuration for hybrid attention mechanism."""
    use_flash_attention: bool = True
    use_memory_efficient: bool = True
    head_dim: int = 64
    num_heads: int = 32
    sliding_window: Optional[int] = None
    attention_dropout: float = 0.1


class HybridAttention(LlamaAttention):
    """Hybrid attention implementation combining H2O and transformer attention."""


    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        config: Optional[AttentionConfig] = None
    ):
        """Initialize hybrid attention.

        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            config: Optional attention configuration
        """
        super().__init__(hidden_size, num_attention_heads)
        self.attention_config = config or AttentionConfig()
        self.cache = DynamicCache()
        self.layer_idx = 0


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass for hybrid attention.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            past_key_value: Optional cached key/value states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key/value states
            cache_position: Optional position for cache lookup

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
