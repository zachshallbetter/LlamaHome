"""Tests for H2O cache implementations."""

import pytest
import torch

from src.core.h2o.cache import Cache, HHCache, StaticCache


def test_base_cache():
    """Test base Cache class."""
    cache = Cache(max_length=10)
    assert cache.get_max_length() == 10
    
    with pytest.raises(NotImplementedError):
        cache.get_seq_length()
        
    with pytest.raises(NotImplementedError):
        cache.update(None, None, 0)
        
    with pytest.raises(NotImplementedError):
        cache.update_slimming(None, 1, 0)


def test_hh_cache_init():
    """Test HHCache initialization."""
    window_length = 5
    num_heavy_hitters = 3
    cache = HHCache(window_length, num_heavy_hitters)
    
    assert cache.window_length == window_length
    assert cache.num_heavy_hitters == num_heavy_hitters
    assert cache.get_max_length() == window_length
    assert len(cache.cache) == 0


def test_hh_cache_update():
    """Test HHCache update functionality."""
    cache = HHCache(5, 3)
    
    # Create sample tensors
    key_states = torch.randn(2, 4, 3, 8)  # [batch, heads, seq_len, dim]
    value_states = torch.randn(2, 4, 3, 8)
    layer_idx = 0
    
    # Test initial update
    new_keys, new_values = cache.update(key_states, value_states, layer_idx)
    assert torch.equal(new_keys, key_states)
    assert torch.equal(new_values, value_states)
    assert cache.get_seq_length() == 3
    
    # Test concatenation
    key_states2 = torch.randn(2, 4, 2, 8)
    value_states2 = torch.randn(2, 4, 2, 8)
    new_keys, new_values = cache.update(key_states2, value_states2, layer_idx)
    
    assert new_keys.size(2) == 5  # 3 + 2
    assert new_values.size(2) == 5


def test_static_cache():
    """Test StaticCache functionality."""
    cache = StaticCache(max_length=10)
    
    # Create sample tensors
    key_states = torch.randn(2, 4, 3, 8)
    value_states = torch.randn(2, 4, 3, 8)
    layer_idx = 0
    
    # Test update
    new_keys, new_values = cache.update(key_states, value_states, layer_idx)
    assert torch.equal(new_keys, key_states)
    assert torch.equal(new_values, value_states)
    
    # Test sequence length
    assert cache.get_seq_length() == 3
    
    # Test slimming (should be no-op)
    cache.update_slimming(torch.randn(2, 4, 3), 1, layer_idx)
    assert cache.get_seq_length() == 3


def test_hh_cache_slimming():
    """Test HHCache slimming functionality."""
    cache = HHCache(5, 2)  # Keep top 2 tokens
    
    # Initial states
    key_states = torch.randn(2, 4, 3, 8)
    value_states = torch.randn(2, 4, 3, 8)
    layer_idx = 0
    
    cache.update(key_states, value_states, layer_idx)
    
    # Simulate attention weights where middle token has highest score
    attn_weights = torch.tensor([[[0.1, 0.8, 0.1]]])
    cache.update_slimming(attn_weights, 1, layer_idx)
    
    # Should keep only 2 tokens (num_heavy_hitters)
    assert cache.get_seq_length(layer_idx) == 2


def test_hh_cache_from_legacy():
    """Test HHCache initialization from legacy format."""
    past_key_values = [
        (torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8)),
        (torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8))
    ]
    
    cache = HHCache.from_legacy_cache(5, 3, past_key_values)
    assert len(cache.cache) == 2
    assert cache.get_seq_length(0) == 3
