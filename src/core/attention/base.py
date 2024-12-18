"""Base attention mechanism implementation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from ..utils import LogManager, LogTemplates

class BaseAttention(nn.Module):
    """Base class for attention mechanisms."""

    def __init__(self, hidden_size: int, num_attention_heads: int):
        """Initialize base attention.

        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
        self.logger.info(f"Initializing BaseAttention with hidden_size={hidden_size} and num_attention_heads={num_attention_heads}")

        if self.head_dim * num_attention_heads != hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def _project_states(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to query, key, and value states."""
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states

    def _reshape_states(self, states: torch.Tensor, batch_size: int, seq_length: int) -> torch.Tensor:
        """Reshape states for multi-head attention."""
        return states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)

    def _attn(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and output."""
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query_states, key_states)

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value_states)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass for base attention.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            past_key_value: Optional cached key/value states
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, past_key_value, attention_weights)
        """
        batch_size, seq_length, _ = hidden_states.size()

        query_states, key_states, value_states = self._project_states(hidden_states)
        query_states = self._reshape_states(query_states, batch_size, seq_length)
        key_states = self._reshape_states(key_states, batch_size, seq_length)
        value_states = self._reshape_states(value_states, batch_size, seq_length)

        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value, attn_weights if output_attentions else None
