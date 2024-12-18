"""
Attention management for LlamaHome models.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn

from .base import BaseAttention, MultiHeadAttention, SelfAttention, CrossAttention
from .hybrid import HybridAttention

class AttentionManager:
    """Manages different attention mechanisms for LlamaHome models."""

    def __init__(self, attention_type: str, **kwargs):
        self.attention_type = attention_type
        self.attention = self._initialize_attention(attention_type, **kwargs)

    def _initialize_attention(self, attention_type: str, **kwargs) -> nn.Module:
        """Initialize the specified attention mechanism."""
        attention_classes = {
            'HybridAttention': HybridAttention,
            'BaseAttention': BaseAttention,
            'MultiHeadAttention': MultiHeadAttention,
            'SelfAttention': SelfAttention,
            'CrossAttention': CrossAttention
        }
        
        if attention_type in attention_classes:
            return attention_classes[attention_type](**kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the selected attention mechanism."""
        return self.attention(*args, **kwargs)

    def set_attention_type(self, attention_type: str, **kwargs) -> None:
        """Set a new attention type and reinitialize the attention mechanism."""
        self.attention_type = attention_type
        self.attention = self._initialize_attention(attention_type, **kwargs)
