"""Tests for training data management."""

import pytest
import torch
from typing import Dict, Any

from src.training.data import DataManager, DatasetProcessor, DataConfig
from src.training.cache import CacheManager
from src.training.batch import BatchGenerator
from src.training.augmentation import DataAugmenter

from src.core.config import ConfigManager


@pytest.fixture
def mock_config():
    """Create mock data configuration."""
    return {
        "data": {
            "batch_size": 32,
            "max_sequence_length": 512,
            "num_workers": 4,
            "cache_size": "2GB",
            "validation_split": 0.1,
            "shuffle_buffer_size": 10000,
        },
        "preprocessing": {
            "tokenizer": "llama",
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
        },
        "augmentation": {
            "enabled": True,
            "techniques": ["random_mask", "token_replacement"],
            "mask_probability": 0.15,
        },
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment with sample data."""
    # Create data directories
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True)

    # Create sample training file
    train_file = data_dir / "train.jsonl"
    with open(train_file, "w") as f:
        f.write('{"text": "Sample training text 1", "label": 1}\n')
        f.write('{"text": "Sample training text 2", "label": 0}\n')

    return tmp_path


class TestDataManager:
    """Test suite for data management functionality."""

    def test_dataset_loading(self, mock_config, setup_test_env):
        """Test dataset loading and initialization."""
        data_manager = DataManager(config=mock_config)

        # Test loading from file
        dataset = data_manager.load_dataset(setup_test_env / "data" / "train.jsonl")
        assert len(dataset) == 2
        assert "text" in dataset[0]
        assert "label" in dataset[0]

    def test_data_splitting(self, mock_config, setup_test_env):
        """Test dataset splitting functionality."""
        data_manager = DataManager(config=mock_config)
        dataset = data_manager.load_dataset(setup_test_env / "data" / "train.jsonl")

        # Split dataset
        train_dataset, val_dataset = data_manager.split_dataset(
            dataset, validation_split=mock_config["data"]["validation_split"]
        )

        assert len(train_dataset) + len(val_dataset) == len(dataset)

    def test_data_validation(self, mock_config):
        """Test data validation functionality."""
        data_manager = DataManager(config=mock_config)

        # Test valid data
        valid_sample = {"text": "Valid text", "label": 1}
        assert data_manager.validate_sample(valid_sample)

        # Test invalid data
        invalid_samples = [
            {"text": "", "label": 1},  # Empty text
            {"text": "Text", "label": -1},  # Invalid label
            {"text": None, "label": 1},  # None text
        ]

        for sample in invalid_samples:
            assert not data_manager.validate_sample(sample)

    def test_data_statistics(self, mock_config, setup_test_env):
        """Test data statistics computation."""
        data_manager = DataManager(config=mock_config)
        dataset = data_manager.load_dataset(setup_test_env / "data" / "train.jsonl")

        # Compute statistics
        stats = data_manager.compute_statistics(dataset)

        assert "num_samples" in stats
        assert "avg_length" in stats
        assert "label_distribution" in stats


class TestDatasetProcessor:
    """Test suite for dataset processing functionality."""

    def test_tokenization(self, mock_config):
        """Test text tokenization."""
        processor = DatasetProcessor(config=mock_config)

        # Test tokenization
        text = "Sample text for tokenization"
        tokens = processor.tokenize(text)

        assert isinstance(tokens, dict)
        assert "input_ids" in tokens
        assert "attention_mask" in tokens

    def test_preprocessing_pipeline(self, mock_config):
        """Test complete preprocessing pipeline."""
        processor = DatasetProcessor(config=mock_config)

        # Test preprocessing steps
        sample = {"text": "Sample text for preprocessing", "label": 1}

        processed = processor.preprocess(sample)
        assert "input_ids" in processed
        assert "attention_mask" in processed
        assert "label" in processed

    def test_special_token_handling(self, mock_config):
        """Test special token handling in preprocessing."""
        processor = DatasetProcessor(config=mock_config)

        # Test with special tokens
        text = "Text with [MASK] token"
        tokens = processor.tokenize(text)

        assert processor.tokenizer.mask_token_id in tokens["input_ids"]

    def test_sequence_length_handling(self, mock_config):
        """Test sequence length handling."""
        processor = DatasetProcessor(config=mock_config)

        # Test long sequence
        long_text = " ".join(["word"] * 1000)
        tokens = processor.tokenize(long_text)

        assert len(tokens["input_ids"]) <= mock_config["data"]["max_sequence_length"]


class TestCacheManager:
    """Test suite for data caching functionality."""

    def test_cache_initialization(self, mock_config, setup_test_env):
        """Test cache initialization."""
        cache_manager = CacheManager(
            cache_dir=setup_test_env / ".cache", config=mock_config
        )

        assert cache_manager.cache_size == "2GB"
        assert cache_manager.cache_dir.exists()

    def test_data_caching(self, mock_config, setup_test_env):
        """Test data caching functionality."""
        cache_manager = CacheManager(
            cache_dir=setup_test_env / ".cache", config=mock_config
        )

        # Cache some data
        data = {"key": torch.randn(100, 100)}
        cache_key = "test_data"

        cache_manager.cache_data(cache_key, data)
        assert cache_manager.is_cached(cache_key)

        # Retrieve cached data
        retrieved_data = cache_manager.get_cached_data(cache_key)
        assert torch.equal(retrieved_data["key"], data["key"])

    def test_cache_eviction(self, mock_config, setup_test_env):
        """Test cache eviction policy."""
        cache_manager = CacheManager(
            cache_dir=setup_test_env / ".cache", config=mock_config
        )

        # Fill cache
        large_data = {"key": torch.randn(10000, 10000)}
        cache_manager.cache_data("large_data", large_data)

        # Try to cache more data
        more_data = {"key": torch.randn(10000, 10000)}
        cache_manager.cache_data("more_data", more_data)

        # Verify cache size is maintained
        assert cache_manager.get_cache_size() <= cache_manager.get_max_cache_size()


class TestBatchGenerator:
    """Test suite for batch generation functionality."""

    def test_batch_creation(self, mock_config):
        """Test batch creation from dataset."""
        batch_generator = BatchGenerator(config=mock_config)

        # Create sample dataset
        dataset = [
            {"input_ids": torch.randint(0, 1000, (100,)), "label": i}
            for i in range(100)
        ]

        # Generate batch
        batch = batch_generator.generate_batch(dataset[:32])
        assert batch["input_ids"].shape[0] == mock_config["data"]["batch_size"]

    def test_dynamic_batching(self, mock_config):
        """Test dynamic batch size adjustment."""
        batch_generator = BatchGenerator(config=mock_config)

        # Test with different sequence lengths
        sequences = [
            {"input_ids": torch.randint(0, 1000, (length,))}
            for length in [50, 100, 150, 200]
        ]

        batch = batch_generator.generate_dynamic_batch(sequences)
        assert all(
            len(seq["input_ids"]) == len(batch["input_ids"][0]) for seq in sequences
        )

    def test_batch_collation(self, mock_config):
        """Test batch collation functionality."""
        batch_generator = BatchGenerator(config=mock_config)

        # Create samples with different lengths
        samples = [
            {
                "input_ids": torch.randint(0, 1000, (length,)),
                "attention_mask": torch.ones(length),
                "label": i % 2,
            }
            for i, length in enumerate([50, 75, 100])
        ]

        # Test collation
        batch = batch_generator.collate_batch(samples)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert all(tensor.shape[0] == len(samples) for tensor in batch.values())


class TestDataAugmenter:
    """Test suite for data augmentation functionality."""

    def test_random_masking(self, mock_config):
        """Test random masking augmentation."""
        augmenter = DataAugmenter(config=mock_config)

        # Test masking
        input_ids = torch.randint(0, 1000, (100,))
        masked_ids = augmenter.random_mask(input_ids)

        assert not torch.equal(input_ids, masked_ids)
        assert (masked_ids == augmenter.mask_token_id).sum() > 0

    def test_token_replacement(self, mock_config):
        """Test token replacement augmentation."""
        augmenter = DataAugmenter(config=mock_config)

        # Test replacement
        input_ids = torch.randint(0, 1000, (100,))
        augmented_ids = augmenter.token_replacement(input_ids)

        assert not torch.equal(input_ids, augmented_ids)

    def test_augmentation_pipeline(self, mock_config):
        """Test complete augmentation pipeline."""
        augmenter = DataAugmenter(config=mock_config)

        # Test pipeline
        sample = {
            "input_ids": torch.randint(0, 1000, (100,)),
            "attention_mask": torch.ones(100),
            "label": 1,
        }

        augmented = augmenter.augment(sample)
        assert "input_ids" in augmented
        assert "attention_mask" in augmented
        assert "label" in augmented
        assert not torch.equal(augmented["input_ids"], sample["input_ids"])
