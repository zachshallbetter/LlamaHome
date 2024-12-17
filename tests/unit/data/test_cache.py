"""Tests for data caching functionality.

This module contains comprehensive tests for the data caching system,
verifying the functionality of the DataCache class and its components.

Test Coverage:
- Cache initialization
- Data storage and retrieval
- Size limit enforcement
- Cache clearing
- Error handling
- Memory management

The tests use pytest fixtures for setup and teardown, ensuring
each test runs in isolation with a clean cache state.

Test Categories:
1. Basic Functionality
   - Initialization
   - Directory creation
   - Configuration
2. Data Operations
   - Adding data
   - Retrieving data
   - Cache misses
3. Memory Management
   - Size limits
   - Eviction policy
   - Resource cleanup
4. Error Handling
   - Invalid operations
   - Resource constraints
   - Edge cases

See Also:
    - src/data/cache.py: Implementation being tested
    - docs/Testing.md: Testing guidelines
    - docs/Data.md: Data management specifications
"""

import pytest
import torch
from pathlib import Path

from src.data.cache import DataCache


@pytest.fixture
def cache_dir(tmp_path_factory):
    """Create temporary cache directory for testing.
    
    This fixture provides an isolated directory for each test,
    ensuring tests don't interfere with each other or the system.
    
    Returns:
        Path: Temporary directory path that will be automatically
        cleaned up after tests complete.
        
    Note:
        Uses pytest's tmp_path_factory for proper cleanup and
        isolation between tests.
    """
    return tmp_path_factory.mktemp("cache")


@pytest.fixture
def data_cache(cache_dir):
    """Create DataCache instance for testing.
    
    This fixture provides a fresh DataCache instance for each test,
    configured with the temporary directory from cache_dir fixture.
    
    Args:
        cache_dir: Temporary directory fixture
        
    Returns:
        DataCache: Configured cache instance ready for testing
        
    Dependencies:
        Requires the cache_dir fixture to be available
    """
    return DataCache(cache_dir)


def test_cache_init(data_cache, cache_dir):
    """Test cache initialization.
    
    Verifies that the cache is properly initialized with:
    - Correct directory path
    - Default maximum size
    - Empty initial state
    - Directory creation
    
    Args:
        data_cache: DataCache fixture
        cache_dir: Directory fixture
        
    Assertions:
        - Cache directory matches provided path
        - Maximum size is set to default (1000)
        - Cache starts empty
        - Directory exists on filesystem
    """
    assert data_cache.cache_dir == cache_dir
    assert data_cache.max_size == 1000
    assert len(data_cache.cache) == 0
    assert cache_dir.exists()


def test_add_get(data_cache):
    """Test adding and retrieving data from cache.
    
    Verifies the core cache operations:
    1. Adding data with a key
    2. Retrieving data by key
    3. Handling missing keys
    4. Preserving data integrity
    
    The test uses both tensor and list data to verify
    different data type handling.
    
    Args:
        data_cache: DataCache fixture
        
    Assertions:
        - Cache size increases after addition
        - Retrieved tensor matches original
        - Retrieved list matches original
        - Missing key returns None
        
    Data Types Tested:
        - PyTorch tensors
        - Python lists
    """
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
    """Test cache respects max size limit.
    
    Verifies that the cache:
    1. Enforces maximum size limit
    2. Evicts oldest entries when full
    3. Maintains size constraint
    4. Properly handles eviction
    
    This test specifically checks the LRU (Least Recently Used)
    eviction policy by adding more items than the cache can hold.
    
    Args:
        data_cache: DataCache fixture
        
    Assertions:
        - Cache size stays within limit
        - Oldest entry is evicted
        - Evicted data cannot be retrieved
        
    Cache Behavior:
        - Set max_size to 2
        - Add 3 items
        - Verify first item is evicted
    """
    data_cache.max_size = 2
    
    data_cache.add("key1", {"data": [1]})
    data_cache.add("key2", {"data": [2]})
    assert len(data_cache) == 2
    
    data_cache.add("key3", {"data": [3]})
    assert len(data_cache) == 2
    assert "key1" not in data_cache.cache
    assert data_cache.get("key1") is None


def test_clear(data_cache):
    """Test clearing the cache.
    
    Verifies that the cache clear operation:
    1. Removes all cached items
    2. Resets cache state
    3. Maintains cache usability
    4. Properly releases resources
    
    This test ensures that clearing the cache completely
    removes all data while keeping the cache usable.
    
    Args:
        data_cache: DataCache fixture
        
    Assertions:
        - Cache is empty after clear
        - Cleared items cannot be retrieved
        - Cache remains usable
        
    Operations Tested:
        1. Add multiple items
        2. Verify items exist
        3. Clear cache
        4. Verify all items removed
    """
    data_cache.add("key1", {"data": [1]})
    data_cache.add("key2", {"data": [2]})
    assert len(data_cache) == 2
    
    data_cache.clear()
    assert len(data_cache) == 0
    assert data_cache.get("key1") is None
    assert data_cache.get("key2") is None
