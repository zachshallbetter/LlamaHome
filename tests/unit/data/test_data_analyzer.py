"""Tests for data analysis functionality."""

import pytest
from pathlib import Path
from typing import Dict, Any

from src.data.processing.analyzer import DataAnalyzer
from src.core.config.log import LogConfig


@pytest.fixture
def mock_config():
    """Create mock analysis configuration."""
    return {
        "analysis": {
            "batch_size": 32,
            "max_sequence_length": 512,
            "num_workers": 4,
            "cache_size": "2GB",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_output": False,
        }
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment."""
    # Create necessary directories
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    
    return tmp_path


class TestDataAnalyzer:
    """Test suite for data analysis functionality."""

    def test_analyzer_initialization(self, mock_config):
        """Test analyzer initialization."""
        log_config = LogConfig(**mock_config["logging"])
        analyzer = DataAnalyzer(config=mock_config["analysis"], log_config=log_config)
        
        assert analyzer.batch_size == mock_config["analysis"]["batch_size"]
        assert analyzer.max_sequence_length == mock_config["analysis"]["max_sequence_length"]

    def test_data_statistics(self, mock_config, setup_test_env):
        """Test data statistics computation."""
        log_config = LogConfig(**mock_config["logging"])
        analyzer = DataAnalyzer(config=mock_config["analysis"], log_config=log_config)
        
        # Create sample data
        data = [
            {"text": "Sample text 1", "label": 1},
            {"text": "Sample text 2", "label": 0},
            {"text": "Sample text 3", "label": 1},
        ]
        
        stats = analyzer.compute_statistics(data)
        assert "num_samples" in stats
        assert "avg_length" in stats
        assert "label_distribution" in stats

    def test_data_validation(self, mock_config):
        """Test data validation functionality."""
        log_config = LogConfig(**mock_config["logging"])
        analyzer = DataAnalyzer(config=mock_config["analysis"], log_config=log_config)
        
        # Test valid data
        valid_data = [
            {"text": "Valid text", "label": 1},
            {"text": "Another valid text", "label": 0},
        ]
        assert analyzer.validate_data(valid_data)
        
        # Test invalid data
        invalid_data = [
            {"text": "", "label": 1},  # Empty text
            {"text": "Valid", "label": -1},  # Invalid label
            {"text": None, "label": 1},  # None text
        ]
        assert not analyzer.validate_data(invalid_data)

    def test_data_preprocessing(self, mock_config):
        """Test data preprocessing functionality."""
        log_config = LogConfig(**mock_config["logging"])
        analyzer = DataAnalyzer(config=mock_config["analysis"], log_config=log_config)
        
        # Test preprocessing
        data = [
            {"text": "Sample text", "label": 1},
            {"text": "Another text", "label": 0},
        ]
        
        processed = analyzer.preprocess_data(data)
        assert len(processed) == len(data)
        assert all("processed_text" in item for item in processed)

    def test_error_handling(self, mock_config):
        """Test error handling in analyzer."""
        log_config = LogConfig(**mock_config["logging"])
        analyzer = DataAnalyzer(config=mock_config["analysis"], log_config=log_config)
        
        # Test with invalid input
        with pytest.raises(ValueError):
            analyzer.compute_statistics(None)
            
        with pytest.raises(ValueError):
            analyzer.validate_data(None)
            
        with pytest.raises(ValueError):
            analyzer.preprocess_data(None)

    def test_caching(self, mock_config, setup_test_env):
        """Test caching functionality."""
        log_config = LogConfig(**mock_config["logging"])
        analyzer = DataAnalyzer(
            config=mock_config["analysis"],
            log_config=log_config,
            cache_dir=setup_test_env / ".cache"
        )
        
        # Test data caching
        data = [{"text": "Cache test", "label": 1}]
        cache_key = "test_cache"
        
        analyzer.cache_data(cache_key, data)
        assert analyzer.is_cached(cache_key)
        
        retrieved = analyzer.get_cached_data(cache_key)
        assert retrieved == data 