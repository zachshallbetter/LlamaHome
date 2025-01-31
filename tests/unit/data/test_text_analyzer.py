"""Tests for text analysis functionality."""

import logging
from pathlib import Path
from unittest.mock import patch

import nltk
import pytest
from rich.console import Console

from src.data.processing.analyzer import TextAnalyzer, AnalysisConfig
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