"""Tests for text analysis functionality.

This module contains test cases for the TextAnalyzer class and related functionality.
Tests cover initialization, readability metrics, POS tagging, semantic features,
document processing and error handling.

@see src/data/analyzer.py
"""

import logging
from pathlib import Path
from unittest.mock import patch

import nltk
import pytest
from rich.console import Console

from src.data.processing.analyzer import DataAnalyzer, TextAnalyzer, AnalysisConfig, create_analyzer
from src.core.utils import LogManager, LogTemplates
from src.core.config.log import LogConfig
# Set up console for test output
console = Console()


@pytest.fixture(autouse=True)
def setup_logging(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Configure logging for tests."""
    log_config = LogConfig()
    log_config.BASE_DIR = tmp_path / ".logs"
    log_manager = LogManager(log_config)
    logger = log_manager.get_logger("text_analyzer", log_type="app", subdir="debug")
    logger.setLevel(logging.WARNING)
    yield
    log_manager.cleanup_logs()


@pytest.fixture(autouse=True)
def setup_nltk() -> None:
    """Ensure required NLTK data is available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    yield


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return """
    The quick brown fox jumps over the lazy dog. This is a test sentence.
    Here is another sentence with some more complexity. And here's one with
    contractions and punctuation! Finally, we have a question?
    """


@pytest.fixture
def config() -> AnalysisConfig:
    """Create a test configuration."""
    return AnalysisConfig(
        enable_pos_tagging=True,
        enable_readability_metrics=True,
        min_sentence_length=3,
        max_sentence_length=100
    )


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


class TestTextAnalyzer:
    """Test cases for TextAnalyzer."""

    def test_initialization(self, config: AnalysisConfig) -> None:
        """Test analyzer initialization."""
        analyzer = TextAnalyzer(config)
        assert analyzer.config == config

    def test_calculate_readability_metrics(self, config: AnalysisConfig, sample_text: str) -> None:
        """Test readability metrics calculation."""
        metrics = config.calculate_readability_metrics(sample_text)
        assert "ari" in metrics
        assert "cli" in metrics
        assert "pos_distribution" in metrics
        assert isinstance(metrics["ari"], float)
        assert isinstance(metrics["cli"], float)
        assert 0 <= metrics["pos_distribution"] <= 1

    def test_calculate_readability_metrics_disabled(self, sample_text: str) -> None:
        """Test readability metrics when disabled."""
        config = AnalysisConfig(enable_readability_metrics=False)
        analyzer = TextAnalyzer(config)
        metrics = analyzer.calculate_readability_metrics(sample_text)
        assert metrics == {}

    def test_calculate_pos_distribution(self, config: AnalysisConfig, sample_text: str) -> None:
        """Test POS distribution calculation."""
        distribution = config._calculate_pos_distribution(sample_text)
        assert isinstance(distribution, float)
        assert 0 <= distribution <= 1

    def test_calculate_pos_distribution_disabled(self, sample_text: str) -> None:
        """Test POS distribution when disabled."""
        config = AnalysisConfig(enable_pos_tagging=False)
        analyzer = TextAnalyzer(config)
        distribution = analyzer._calculate_pos_distribution(sample_text)
        assert distribution == 0.0

    def test_calculate_ari(self, config: AnalysisConfig, sample_text: str) -> None:
        """Test Automated Readability Index calculation."""
        ari = config._calculate_ari(sample_text)
        assert isinstance(ari, float)
        assert ari > 0

    def test_calculate_ari_edge_cases(self, config: AnalysisConfig) -> None:
        """Test ARI calculation with edge cases."""
        assert config._calculate_ari("") == 0.0
        assert config._calculate_ari(" ") == 0.0
        assert config._calculate_ari("a") == 0.0
        assert config._calculate_ari("a.") == 0.0

    def test_calculate_cli(self, config: AnalysisConfig, sample_text: str) -> None:
        """Test Coleman-Liau Index calculation."""
        cli = config._calculate_cli(sample_text)
        assert isinstance(cli, float)

    def test_calculate_cli_edge_cases(self, config: AnalysisConfig) -> None:
        """Test CLI calculation with edge cases."""
        assert config._calculate_cli("") == 0.0
        assert config._calculate_cli(" ") == 0.0
        assert config._calculate_cli("a") == 0.0
        assert config._calculate_cli("a.") == 0.0

    def test_get_semantic_features(self, config: AnalysisConfig, sample_text: str) -> None:
        """Test semantic feature extraction."""
        features = config._get_semantic_features(sample_text)
        assert "pos_distribution" in features
        assert "readability" in features
        assert isinstance(features["pos_distribution"], dict)
        assert isinstance(features["readability"], dict)
        assert len(features["pos_distribution"]) > 0
        assert "automated_readability_index" in features["readability"]
        assert "coleman_liau_index" in features["readability"]

    def test_get_semantic_features_disabled(self, sample_text: str) -> None:
        """Test semantic features when disabled."""
        config = AnalysisConfig(
            enable_pos_tagging=False,
            enable_readability_metrics=False
        )
        analyzer = TextAnalyzer(config)
        features = analyzer._get_semantic_features(sample_text)
        assert features == {}

    def test_get_semantic_features_error_handling(self, config: AnalysisConfig, caplog: pytest.LogCaptureFixture) -> None:
        """Test error handling in semantic feature extraction."""
        with patch("nltk.pos_tag", side_effect=Exception("NLTK error")):
            features = config._get_semantic_features("test text")
            assert features == {}
            assert any("NLTK error" in record.message for record in caplog.records)

    def test_validate_document_format(self, config: AnalysisConfig, tmp_path: Path) -> None:
        """Test document format validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        validation = config.validate_document_format(str(test_file))
        assert isinstance(validation, dict)
        assert validation["exists"] is True
        assert validation["is_file"] is True
        assert validation["size"] > 0
        assert validation["extension"] == ".txt"

    def test_extract_and_clean_text(self, config: AnalysisConfig, tmp_path: Path) -> None:
        """Test text extraction and cleaning."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        validation = config.validate_document_format(str(test_file))
        text = config.extract_and_clean_text(str(test_file), validation)
        assert text == "Test content"


@pytest.mark.integration
class TestTextAnalyzerIntegration:
    """Integration tests for TextAnalyzer."""

    @pytest.fixture(autouse=True)
    def setup_test_files(self, tmp_path: Path) -> None:
        """Set up test files for integration tests."""
        self.test_dir = tmp_path / "test_docs"
        self.test_dir.mkdir()

        # Create test documents
        docs = {
            "short.txt": "Short test document.",
            "medium.txt": "Medium length document with some complexity.",
            "long.txt": """Long document with multiple sentences and paragraphs.
            This tests the system's ability to handle larger texts.
            Including various punctuation marks! And questions?"""
        }

        for name, content in docs.items():
            (self.test_dir / name).write_text(content)

        yield

        # Cleanup
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()

    def test_end_to_end_processing(self, config: AnalysisConfig) -> None:
        """Test end-to-end document processing."""
        analyzer = create_analyzer(config)

        for doc in self.test_dir.glob("*.txt"):
            result = analyzer.process_document(str(doc))

            assert "validation" in result
            assert "content" in result
            assert isinstance(result["validation"], dict)
            assert isinstance(result["content"], str)
            assert result["validation"]["exists"] is True
            assert result["validation"]["is_file"] is True
            assert result["validation"]["size"] > 0
            assert result["validation"]["extension"] == ".txt"
            assert len(result["content"]) > 0

    def test_process_with_different_configs(self, config: AnalysisConfig) -> None:
        """Test processing with different configurations."""
        configs = [
            AnalysisConfig(),  # Default config
            AnalysisConfig(enable_pos_tagging=False),  # No POS tagging
            AnalysisConfig(enable_readability_metrics=False)  # No readability metrics
        ]

        for config in configs:
            analyzer = TextAnalyzer(config)
            for doc in self.test_dir.glob("*.txt"):
                result = analyzer.process_document(str(doc))
                assert "validation" in result
                assert "content" in result
                assert isinstance(result["content"], str)
                assert len(result["content"]) > 0

    def test_error_handling(self) -> None:
        """Test error handling for various scenarios."""
        analyzer = TextAnalyzer(AnalysisConfig())

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            analyzer.process_document(str(self.test_dir / "nonexistent.txt"))

        # Test with empty file
        empty_file = self.test_dir / "empty.txt"
        empty_file.write_text("")
        result = analyzer.process_document(str(empty_file))
        assert result["content"] == ""

        # Test with file containing only whitespace
        whitespace_file = self.test_dir / "whitespace.txt"
        whitespace_file.write_text("   \n   \t   ")
        result = analyzer.process_document(str(whitespace_file))
        assert result["content"] == ""


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
