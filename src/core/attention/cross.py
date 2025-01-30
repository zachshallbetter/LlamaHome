"""Cross attention mechanism implementation."""

from typing import Optional, Tuple

import torch

from .base import BaseAttention


class CrossAttention(BaseAttention):
    """Cross attention mechanism."""

    def __init__(self, hidden_size: int, num_attention_heads: int):
        """Initialize cross attention.

        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
        """
        super().__init__(hidden_size, num_attention_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass for cross attention.

        Args:
            hidden_states: Input tensor
            encoder_hidden_states: Encoder hidden states for cross attention
            attention_mask: Optional attention mask
            past_key_value: Optional cached key/value states
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, past_key_value, attention_weights)
        """
        batch_size, seq_length, _ = hidden_states.size()
        _, encoder_seq_length, _ = encoder_hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = self._reshape_states(query_states, batch_size, seq_length)
        key_states = self._reshape_states(key_states, batch_size, encoder_seq_length)
        value_states = self._reshape_states(
            value_states, batch_size, encoder_seq_length
        )

        attn_output, attn_weights = self._attn(
            query_states, key_states, value_states, attention_mask
        )
        attn_output = attn_output.contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value, attn_weights if output_attentions else None
