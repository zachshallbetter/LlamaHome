"""Tests for hybrid attention mechanism combining H2O and transformer attention."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.core.attention import HybridAttention
from src.core.config_handler import ConfigManager


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = MagicMock()
    config.attention = {
        "window_length": 1024,
        "heavy_hitter_tokens": 256,
        "use_flash_attention": True,
        "use_memory_efficient": True
    }
    return config


@pytest.fixture
def mock_base_attention():
    """Create mock base attention module."""
    attention = MagicMock()
    attention.forward.return_value = (
        torch.randn(2, 8, 32, 64),  # attention output
        torch.softmax(torch.randn(2, 8, 32, 32), dim=-1)  # attention weights
    )
    return attention


class TestHybridAttention:
    """Test suite for hybrid attention implementation."""
    
    def test_initialization(self, mock_config):
        """Test attention module initialization."""
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        assert attention.hidden_size == 1024
        assert attention.num_attention_heads == 8
        assert attention.window_length == mock_config.attention["window_length"]
        assert attention.heavy_hitter_tokens == mock_config.attention["heavy_hitter_tokens"]
    
    def test_attention_computation(self, mock_config, mock_base_attention):
        """Test attention computation combining H2O and transformer attention."""
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        # Mock input tensors
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        attention_mask = torch.ones(batch_size, seq_len)
        
        with patch.object(attention, '_compute_h2o_attention') as mock_h2o:
            with patch.object(attention, '_compute_transformer_attention') as mock_transformer:
                # Set up mock returns
                mock_h2o.return_value = torch.randn(batch_size, seq_len, 1024)
                mock_transformer.return_value = torch.randn(batch_size, seq_len, 1024)
                
                # Test forward pass
                output = attention.forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask
                )
                
                # Verify both attention mechanisms were called
                mock_h2o.assert_called_once()
                mock_transformer.assert_called_once()
                
                # Verify output shape
                assert output.shape == (batch_size, seq_len, 1024)
    
    def test_h2o_attention(self, mock_config):
        """Test H2O attention computation."""
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        
        with patch.object(attention, '_get_heavy_hitters') as mock_heavy_hitters:
            # Mock heavy hitter computation
            mock_heavy_hitters.return_value = torch.randint(
                0, seq_len, (batch_size, attention.heavy_hitter_tokens)
            )
            
            # Test H2O attention
            output = attention._compute_h2o_attention(hidden_states)
            
            # Verify output
            assert output.shape == (batch_size, seq_len, 1024)
            mock_heavy_hitters.assert_called_once()
    
    def test_transformer_attention(self, mock_config):
        """Test transformer attention computation."""
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Test transformer attention
        output = attention._compute_transformer_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        # Verify output
        assert output.shape == (batch_size, seq_len, 1024)
    
    def test_attention_mask_handling(self, mock_config):
        """Test attention mask handling in hybrid attention."""
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        
        # Test with different mask types
        masks = [
            torch.ones(batch_size, seq_len),  # Regular mask
            torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(batch_size, -1, -1),  # Causal mask
            None  # No mask
        ]
        
        for mask in masks:
            output = attention.forward(
                hidden_states=hidden_states,
                attention_mask=mask
            )
            assert output.shape == (batch_size, seq_len, 1024)
    
    def test_memory_efficient_mode(self, mock_config):
        """Test memory efficient attention computation."""
        mock_config.attention["use_memory_efficient"] = True
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        
        with patch.object(attention, '_compute_chunked_attention') as mock_chunked:
            mock_chunked.return_value = torch.randn(batch_size, seq_len, 1024)
            
            output = attention.forward(hidden_states=hidden_states)
            
            mock_chunked.assert_called_once()
            assert output.shape == (batch_size, seq_len, 1024)
    
    def test_flash_attention(self, mock_config):
        """Test flash attention when available."""
        mock_config.attention["use_flash_attention"] = True
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            output = attention.forward(hidden_states=hidden_states)
            assert output.shape == (batch_size, seq_len, 1024)
    
    def test_gradient_checkpointing(self, mock_config):
        """Test gradient checkpointing in attention computation."""
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        attention.gradient_checkpointing_enable()
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024, requires_grad=True)
        
        output = attention.forward(hidden_states=hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert hidden_states.grad is not None
    
    def test_attention_dropout(self, mock_config):
        """Test attention dropout functionality."""
        mock_config.attention["dropout"] = 0.1
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        
        # Test in training mode
        attention.train()
        train_output = attention.forward(hidden_states=hidden_states)
        
        # Test in eval mode
        attention.eval()
        eval_output = attention.forward(hidden_states=hidden_states)
        
        # Outputs should be different in training mode due to dropout
        assert not torch.allclose(train_output, eval_output)
    
    def test_attention_caching(self, mock_config):
        """Test attention caching mechanism."""
        attention = HybridAttention(
            hidden_size=1024,
            num_attention_heads=8,
            config=mock_config
        )
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        
        # First forward pass
        output1 = attention.forward(
            hidden_states=hidden_states,
            use_cache=True
        )
        
        # Second forward pass with cached values
        output2 = attention.forward(
            hidden_states=hidden_states,
            use_cache=True,
            past_key_values=attention.get_cache()
        )
        
        assert attention.has_cache()
        assert output2.shape == (batch_size, seq_len, 1024) 