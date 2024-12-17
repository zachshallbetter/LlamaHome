"""Tests for data caching functionality."""

import pytest
import torch
from pathlib import Path

from src.data.cache import DataCache


@pytest.fixture
def cache_dir(tmp_path_factory):
    """Create temporary cache directory."""
    return tmp_path_factory.mktemp("cache")


@pytest.fixture
def data_cache(cache_dir):
    """Create DataCache instance for testing."""
    return DataCache(cache_dir)


def test_cache_init(data_cache, cache_dir):
    """Test cache initialization."""
    assert data_cache.cache_dir == cache_dir
    assert data_cache.max_size == 1000
    assert len(data_cache.cache) == 0
    assert cache_dir.exists()


def test_add_get(data_cache):
    """Test adding and retrieving data from cache."""
    test_data = {
        "tensor": torch.tensor([1, 2, 3]),
        "list": [4, 5, 6]
    }
    
    data_cache.add("test_key", test_data)
    assert len(data_cache) == 1
    
    retrieved = data_cache.get("test_key")
    assert torch.equal(retrieved["tensor"], test_data["tensor"])
    assert retrieved["list"] == test_data["list"]
    
    assert data_cache.get("nonexistent") is None


def test_max_size(data_cache):
    """Test cache respects max size limit."""
    data_cache.max_size = 2
    
    data_cache.add("key1", {"data": [1]})
    data_cache.add("key2", {"data": [2]})
    assert len(data_cache) == 2
    
    data_cache.add("key3", {"data": [3]})
    assert len(data_cache) == 2
    assert "key1" not in data_cache.cache
    assert data_cache.get("key1") is None


def test_clear(data_cache):
    """Test clearing the cache."""
    data_cache.add("key1", {"data": [1]})
    data_cache.add("key2", {"data": [2]})
    assert len(data_cache) == 2
    
    data_cache.clear()
    assert len(data_cache) == 0
    assert data_cache.get("key1") is None
    assert data_cache.get("key2") is None
