"""Tests for H2O model implementation."""

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from src.core.h2o.model import H2OLlamaForCausalLM
from src.core.h2o.cache import HHCache, StaticCache


def test_h2o_model_init():
    """Test H2O model initialization."""
    config = LlamaConfig(
        num_hidden_layers=4,
        num_heavy_hitter_tokens=10,
        num_window_length=100
    )
    model = H2OLlamaForCausalLM(config)

    assert model.model.num_heavy_hitter_tokens == 10
    assert model.model.num_window_length == 100
    assert len(model.model.layers) == 4


def test_prepare_inputs_for_generation():
    """Test input preparation for generation."""
    config = LlamaConfig(
        num_hidden_layers=2,
        num_heavy_hitter_tokens=5,
        num_window_length=50
    )
    model = H2OLlamaForCausalLM(config)

    # Test with no cache
    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    
    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache_position=torch.zeros(1),
    )
    
    assert "input_ids" in inputs
    assert inputs["input_ids"].shape == input_ids.shape
    assert inputs["attention_mask"].shape == attention_mask.shape

    # Test with H2O cache
    cache = HHCache(window_length=50, num_heavy_hitters=5)
    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=cache,
        attention_mask=attention_mask,
        cache_position=torch.zeros(1),
    )
    
    assert "input_ids" in inputs
    assert "past_key_values" in inputs
    assert isinstance(inputs["past_key_values"], HHCache)

    # Test with static cache
    cache = StaticCache(max_length=50)
    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=cache,
        attention_mask=attention_mask,
        cache_position=torch.zeros(1),
    )
    
    assert "input_ids" in inputs
    assert "past_key_values" in inputs
    assert isinstance(inputs["past_key_values"], StaticCache)
