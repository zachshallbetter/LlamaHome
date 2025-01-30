"""Platform-agnostic efficient attention implementation."""

import warnings
from functools import lru_cache
from typing import Any, Dict, Optional

import torch


@lru_cache(maxsize=1)
def get_optimal_attention_backend() -> str:
    """
    Determine the best available attention implementation for the current
    environment.
    """
    if torch.cuda.is_available():
        try:
            import xformers  # noqa: F401

            return "xformers"
        except ImportError:
            return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class EfficientAttention:
    """
    Platform-agnostic efficient attention implementation.
    Automatically selects the best backend based on hardware availability.
    """

    def __init__(self, device_map: Optional[str] = "auto"):
        self.backend = get_optimal_attention_backend()
        self.device_map = device_map

        if self.backend == "xformers":
            from xformers.ops import memory_efficient_attention

            self.attention_func = memory_efficient_attention

        # Log which backend we're using
        warnings.warn(f"Using {self.backend} attention backend", stacklevel=2)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute attention with the most efficient available backend.

        Args:
            query: (batch_size, seq_len, num_heads, head_dim)
            key: (batch_size, seq_len, num_heads, head_dim)
            value: (batch_size, seq_len, num_heads, head_dim)
            mask: Optional attention mask
            dropout_p: Dropout probability
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
            **kwargs: Additional backend-specific arguments
        """
        if scale is None:
            scale = query.shape[-1] ** -0.5

        # Handle different backends
        if self.backend == "xformers":
            # xformers has its own memory efficient implementation
            return self.attention_func(
                query, key, value, attn_bias=mask, p=dropout_p, scale=scale
            )

        elif self.backend in ["cuda", "mps", "cpu"]:
            # Standard scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

            if mask is not None:
                attention_scores = attention_scores.masked_fill(
                    mask == 0, float("-inf")
                )

            attention_probs = torch.softmax(attention_scores, dim=-1)

            if dropout_p > 0.0:
                attention_probs = torch.nn.functional.dropout(
                    attention_probs, p=dropout_p
                )

            return torch.matmul(attention_probs, value)


# Usage example in a transformer layer:
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        device_map = config.get("device_map", "auto")
        self.attention = EfficientAttention(device_map=device_map)
        # ... rest of initialization ...

    def forward(self, query, key, value, mask=None):
        return self.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            dropout_p=self.dropout_p,
            causal=self.causal,
        )
