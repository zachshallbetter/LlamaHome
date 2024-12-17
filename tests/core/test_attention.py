"""Tests for hybrid attention mechanism."""

import pytest
import torch
from torch import nn

from src.core.attention import HybridAttention, AttentionConfig
from src.core.cache import DynamicCache


@pytest.fixture
def config():
    """Create test configuration."""
    class Config:
        def __init__(self):
            self.hidden_size = 512
            self.num_attention_heads = 8
            self.head_dim = 64
            self.use_flash_attention = True
            self.use_memory_efficient = True
            self.sliding_window = None
            self.dropout = 0.1
    return Config()


@pytest.fixture
def attention(config):
    """Create test attention layer."""
    return HybridAttention(config, layer_idx=0)


def test_attention_initialization(attention):
    """Test attention layer initialization."""
    assert isinstance(attention, HybridAttention)
    assert attention.layer_idx == 0
    assert isinstance(attention.cache, DynamicCache)
    assert attention.attention_config.use_flash_attention is True
    assert attention.attention_config.use_memory_efficient is True


def test_memory_efficient_attention(attention):
    """Test memory efficient attention computation."""
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs
    query = torch.randn(batch_size, seq_len, attention.num_heads, attention.head_dim)
    key = torch.randn(batch_size, seq_len, attention.num_heads, attention.head_dim)
    value = torch.randn(batch_size, seq_len, attention.num_heads, attention.head_dim)
    
    # Test without mask
    output, weights = attention._memory_efficient_attention(query, key, value)
    assert output.shape == (batch_size, seq_len, attention.num_heads, attention.head_dim)
    
    # Test with mask
    mask = torch.ones(batch_size, seq_len)
    output, weights = attention._memory_efficient_attention(query, key, value, mask)
    assert output.shape == (batch_size, seq_len, attention.num_heads, attention.head_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attention(attention):
    """Test flash attention computation."""
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs
    query = torch.randn(batch_size, seq_len, attention.num_heads, attention.head_dim).cuda()
    key = torch.randn(batch_size, seq_len, attention.num_heads, attention.head_dim).cuda()
    value = torch.randn(batch_size, seq_len, attention.num_heads, attention.head_dim).cuda()
    
    # Test without mask
    output, weights = attention._flash_attention(query, key, value)
    assert output.shape == (batch_size, seq_len, attention.num_heads, attention.head_dim)
    
    # Test with mask
    mask = torch.ones(batch_size, seq_len).cuda()
    output, weights = attention._flash_attention(query, key, value, mask)
    assert output.shape == (batch_size, seq_len, attention.num_heads, attention.head_dim)


def test_forward_pass(attention):
    """Test forward pass with different configurations."""
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test without cache
    output, past_kv, attn_weights = attention.forward(
        hidden_states,
        attention_mask=attention_mask,
        use_cache=False
    )
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert past_kv is None
    assert attn_weights is None
    
    # Test with cache
    output, past_kv, attn_weights = attention.forward(
        hidden_states,
        attention_mask=attention_mask,
        use_cache=True
    )
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert past_kv is not None
    assert attn_weights is None


def test_sliding_window(config):
    """Test sliding window attention."""
    config.sliding_window = 8
    attention = HybridAttention(config, layer_idx=0)
    
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test forward pass with sliding window
    output, past_kv, attn_weights = attention.forward(
        hidden_states,
        attention_mask=attention_mask,
        use_cache=True
    )
    assert output.shape == (batch_size, seq_len, hidden_size)


def test_cache_management(attention):
    """Test cache management functionality."""
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test cache updates
    output1, past_kv1, _ = attention.forward(hidden_states, use_cache=True)
    assert attention.cache.get_seq_length(0) == seq_len
    
    # Test cache retrieval
    output2, past_kv2, _ = attention.forward(
        hidden_states,
        past_key_value=past_kv1,
        use_cache=True
    )
    assert past_kv2 is not None


def test_attention_dropout(config):
    """Test attention dropout behavior."""
    config.dropout = 0.5
    attention = HybridAttention(config, layer_idx=0)
    
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test in training mode
    attention.train()
    output1 = attention.forward(hidden_states)[0]
    
    # Test in eval mode
    attention.eval()
    output2 = attention.forward(hidden_states)[0]
    
    # Outputs should be different in training mode due to dropout
    assert not torch.allclose(output1, output2)


def test_attention_gradients(attention):
    """Test gradient computation."""
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs requiring gradients
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    
    # Forward pass
    output, _, _ = attention.forward(hidden_states)
    
    # Compute gradients
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()


def test_attention_device_handling(attention):
    """Test device handling."""
    batch_size, seq_len = 2, 16
    hidden_size = 512
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    if torch.cuda.is_available():
        # Move attention to GPU
        attention = attention.cuda()
        hidden_states = hidden_states.cuda()
        
        # Test forward pass on GPU
        output, _, _ = attention.forward(hidden_states)
        assert output.device.type == "cuda"
    else:
        # Test forward pass on CPU
        output, _, _ = attention.forward(hidden_states)
        assert output.device.type == "cpu" 