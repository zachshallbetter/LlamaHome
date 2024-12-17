"""Tests for training cache management system."""

import pytest
import torch
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.cache import (
    TrainingCache,
    CachePolicy,
    MemoryManager,
    CachePersistence,
    DatasetCache
)


@pytest.fixture
def mock_config():
    """Create mock cache configuration."""
    return {
        "cache": {
            "max_size": "4GB",
            "policy": "lru",
            "persistence": True,
            "compression": True,
            "cleanup_threshold": 0.9,
            "preload_datasets": True
        },
        "memory": {
            "target_memory_usage": 0.8,
            "check_interval": 1000,
            "cleanup_margin": 0.1
        }
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment with cache directories."""
    # Create cache directories
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True)
    
    # Create subdirectories
    (cache_dir / "datasets").mkdir()
    (cache_dir / "models").mkdir()
    (cache_dir / "tensors").mkdir()
    
    return tmp_path


class TestTrainingCache:
    """Test suite for main training cache functionality."""
    
    def test_cache_initialization(self, mock_config, setup_test_env):
        """Test cache initialization and configuration."""
        cache = TrainingCache(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        assert cache.max_size == "4GB"
        assert cache.policy == "lru"
        assert cache.persistence_enabled
        assert cache.compression_enabled
    
    def test_data_storage(self, mock_config, setup_test_env):
        """Test data storage functionality."""
        cache = TrainingCache(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        # Store tensor data
        tensor_data = torch.randn(100, 100)
        cache.store("test_tensor", tensor_data, category="tensors")
        
        # Verify storage
        assert cache.exists("test_tensor")
        retrieved = cache.retrieve("test_tensor")
        assert torch.equal(retrieved, tensor_data)
    
    def test_cache_categories(self, mock_config, setup_test_env):
        """Test different cache categories."""
        cache = TrainingCache(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        # Test different categories
        categories = ["datasets", "models", "tensors"]
        for category in categories:
            data = torch.randn(50, 50)
            key = f"test_{category}"
            cache.store(key, data, category=category)
            assert cache.exists(key)
            assert cache.get_category_size(category) > 0
    
    def test_cache_cleanup(self, mock_config, setup_test_env):
        """Test cache cleanup functionality."""
        cache = TrainingCache(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        # Fill cache
        for i in range(100):
            data = torch.randn(1000, 1000)  # Large tensor
            cache.store(f"large_tensor_{i}", data)
        
        # Trigger cleanup
        cache.cleanup()
        
        # Verify cache size is within limits
        assert cache.get_total_size() <= cache.get_max_size()
    
    def test_cache_compression(self, mock_config, setup_test_env):
        """Test data compression functionality."""
        cache = TrainingCache(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        # Store with compression
        data = torch.randn(1000, 1000)
        cache.store("compressed_data", data, compress=True)
        
        # Verify compression
        compressed_size = cache.get_entry_size("compressed_data")
        cache.store("uncompressed_data", data, compress=False)
        uncompressed_size = cache.get_entry_size("uncompressed_data")
        
        assert compressed_size < uncompressed_size


class TestCachePolicy:
    """Test suite for cache eviction policies."""
    
    def test_lru_policy(self, mock_config):
        """Test LRU cache eviction policy."""
        policy = CachePolicy(policy_type="lru", config=mock_config)
        
        # Add entries
        for i in range(10):
            policy.add_entry(f"key_{i}", 100)  # 100 bytes each
        
        # Access some entries
        policy.access_entry("key_0")
        policy.access_entry("key_1")
        
        # Trigger eviction
        evicted = policy.get_entries_to_evict(200)  # Need 200 bytes
        assert "key_0" not in evicted
        assert "key_1" not in evicted
    
    def test_fifo_policy(self, mock_config):
        """Test FIFO cache eviction policy."""
        policy = CachePolicy(policy_type="fifo", config=mock_config)
        
        # Add entries
        for i in range(10):
            policy.add_entry(f"key_{i}", 100)
        
        # Trigger eviction
        evicted = policy.get_entries_to_evict(200)
        assert "key_0" in evicted
        assert "key_1" in evicted
    
    def test_size_based_policy(self, mock_config):
        """Test size-based eviction policy."""
        policy = CachePolicy(policy_type="size", config=mock_config)
        
        # Add entries with different sizes
        policy.add_entry("small", 100)
        policy.add_entry("medium", 200)
        policy.add_entry("large", 300)
        
        # Trigger eviction
        evicted = policy.get_entries_to_evict(200)
        assert "large" in evicted  # Largest entry should be evicted first


class TestMemoryManager:
    """Test suite for memory management functionality."""
    
    def test_memory_tracking(self, mock_config):
        """Test memory usage tracking."""
        memory_manager = MemoryManager(config=mock_config)
        
        # Track memory usage
        with memory_manager.track():
            # Allocate some tensors
            tensors = [torch.randn(1000, 1000) for _ in range(5)]
        
        stats = memory_manager.get_stats()
        assert "peak_memory" in stats
        assert "current_memory" in stats
    
    def test_memory_cleanup(self, mock_config):
        """Test memory cleanup functionality."""
        memory_manager = MemoryManager(config=mock_config)
        
        # Allocate memory
        tensors = [torch.randn(1000, 1000) for _ in range(10)]
        initial_memory = memory_manager.get_current_memory()
        
        # Trigger cleanup
        memory_manager.cleanup()
        final_memory = memory_manager.get_current_memory()
        
        assert final_memory < initial_memory
    
    def test_memory_limits(self, mock_config):
        """Test memory limit enforcement."""
        memory_manager = MemoryManager(config=mock_config)
        
        # Test memory limit checking
        with pytest.raises(MemoryError):
            with memory_manager.enforce_limits():
                # Try to allocate too much memory
                huge_tensor = torch.randn(100000, 100000)


class TestCachePersistence:
    """Test suite for cache persistence functionality."""
    
    def test_cache_saving(self, mock_config, setup_test_env):
        """Test cache state persistence."""
        persistence = CachePersistence(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        # Create cache state
        state = {
            "entries": {
                "key1": {"size": 100, "category": "tensors"},
                "key2": {"size": 200, "category": "datasets"}
            },
            "metadata": {
                "total_size": 300,
                "last_cleanup": "2023-01-01"
            }
        }
        
        # Save state
        persistence.save_state(state)
        
        # Load state
        loaded_state = persistence.load_state()
        assert loaded_state == state
    
    def test_cache_recovery(self, mock_config, setup_test_env):
        """Test cache recovery functionality."""
        persistence = CachePersistence(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        # Simulate cache corruption
        with patch.object(persistence, 'load_state', side_effect=Exception):
            recovered_state = persistence.recover_state()
            assert recovered_state is not None
            assert "entries" in recovered_state
    
    def test_cache_migration(self, mock_config, setup_test_env):
        """Test cache format migration."""
        persistence = CachePersistence(
            cache_dir=setup_test_env / ".cache",
            config=mock_config
        )
        
        # Create old format state
        old_state = {
            "data": {
                "key1": 100,
                "key2": 200
            }
        }
        
        # Migrate to new format
        new_state = persistence.migrate_state(old_state)
        assert "entries" in new_state
        assert "metadata" in new_state


class TestDatasetCache:
    """Test suite for dataset-specific caching."""
    
    def test_dataset_caching(self, mock_config, setup_test_env):
        """Test dataset caching functionality."""
        dataset_cache = DatasetCache(
            cache_dir=setup_test_env / ".cache" / "datasets",
            config=mock_config
        )
        
        # Create sample dataset
        dataset = [
            {"input_ids": torch.randint(0, 1000, (100,))}
            for _ in range(100)
        ]
        
        # Cache dataset
        dataset_cache.cache_dataset("test_dataset", dataset)
        
        # Retrieve dataset
        cached_dataset = dataset_cache.get_dataset("test_dataset")
        assert len(cached_dataset) == len(dataset)
    
    def test_dataset_streaming(self, mock_config, setup_test_env):
        """Test dataset streaming functionality."""
        dataset_cache = DatasetCache(
            cache_dir=setup_test_env / ".cache" / "datasets",
            config=mock_config
        )
        
        # Create large dataset
        large_dataset = [
            {"input_ids": torch.randint(0, 1000, (100,))}
            for _ in range(1000)
        ]
        
        # Test streaming interface
        with dataset_cache.stream_dataset("large_dataset", large_dataset) as stream:
            for batch in stream:
                assert "input_ids" in batch
    
    def test_dataset_preprocessing(self, mock_config, setup_test_env):
        """Test dataset preprocessing in cache."""
        dataset_cache = DatasetCache(
            cache_dir=setup_test_env / ".cache" / "datasets",
            config=mock_config
        )
        
        # Define preprocessing function
        def preprocess(sample):
            return {
                "input_ids": sample["input_ids"],
                "processed": True
            }
        
        # Create dataset
        dataset = [
            {"input_ids": torch.randint(0, 1000, (100,))}
            for _ in range(10)
        ]
        
        # Cache with preprocessing
        dataset_cache.cache_dataset(
            "preprocessed_dataset",
            dataset,
            preprocess_fn=preprocess
        )
        
        # Verify preprocessing
        cached_dataset = dataset_cache.get_dataset("preprocessed_dataset")
        assert all(sample["processed"] for sample in cached_dataset) 