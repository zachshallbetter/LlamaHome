"""Tests for H2O attention mechanism."""

import pytest
import torch
from src.core.h2o.attention import H2OLlamaAttention, attention_forward
from transformers.models.llama.configuration_llama import LlamaConfig


@pytest.fixture
def config():
    """Test config fixture."""
    return LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_heavy_hitter_tokens=4,
        num_window_length=16
    )


def test_h2o_attention_init(config):
    """Test H2O attention initialization."""
    layer_idx = 0
    attention = H2OLlamaAttention(config, layer_idx)
    
    assert attention.layer_idx == layer_idx
    assert attention.past_key_value is None
    assert attention.cache is not None


def test_h2o_attention_forward(config):
    """Test H2O attention forward pass."""
    batch_size = 2
    seq_length = 8
    hidden_size = config.hidden_size
    
    attention = H2OLlamaAttention(config, layer_idx=0)
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    output, past_kv, attentions = attention.forward(
        hidden_states,
        use_cache=True,
        cache_position=torch.tensor([0])
    )
    
    assert output.shape == (batch_size, seq_length, hidden_size)
    assert past_kv is not None
    assert attentions is not None


def test_attention_forward():
    """Test scaled dot product attention."""
    batch_size = 2
    seq_length = 8
    hidden_size = 32
    
    query = torch.randn(batch_size, seq_length, hidden_size)
    key = torch.randn(batch_size, seq_length, hidden_size) 
    value = torch.randn(batch_size, seq_length, hidden_size)
    mask = torch.ones(batch_size, seq_length, seq_length)
    
    output, weights, context = attention_forward(
        query=query,
        key=key,
        value=value,
        mask=mask,
        dropout=0.1
    )
    
    assert output.shape == (batch_size, seq_length, hidden_size)
    assert weights[0].shape == (batch_size, seq_length, seq_length)
    assert context is None
