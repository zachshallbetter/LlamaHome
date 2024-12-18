"""
@module text_analyzer
@description Advanced text analysis system for LlamaHome that performs linguistic analysis,
readability scoring, and text quality assessment. Optimized for local AI model
compatibility including Llama and Stable Diffusion.

@since 1.0.0
@see docs/DATA.md
@see docs/API.md#text-analysis
@see docs/Models.md#text-analysis
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple



try:
    import nltk


    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from ...core.utils import LogManager, LogTemplates



logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class AnalysisConfig:
    """Configuration for text analysis."""


    def __init__(
        self,
        enable_pos_tagging: bool = True,
        enable_readability_metrics: bool = True,
        min_sentence_length: int = 3,
        max_sentence_length: int = 512,
    ) -> None:
        """Initialize analysis configuration.

        Args:
            enable_pos_tagging: Whether to enable POS tagging
            enable_readability_metrics: Whether to calculate readability metrics
            min_sentence_length: Minimum sentence length to process
            max_sentence_length: Maximum sentence length to process
        """
        self.enable_pos_tagging = enable_pos_tagging
        self.enable_readability_metrics = enable_readability_metrics
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length


class TextAnalyzer:
    """Text analysis system."""


    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        """Initialize analyzer.

        Args:
            config: Optional analysis configuration
        """
        self.config = config or AnalysisConfig()

        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')


    def process_document(self, filepath: str) -> Dict[str, Any]:
        """Process document with improved error handling and validation.

        Args:
            filepath: Path to document

        Returns:
            Processed document data

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        validation_result = self.validate_document_format(filepath)
        processed_content = self.extract_and_clean_text(filepath, validation_result)
        return {'validation': validation_result, 'content': processed_content}


    def validate_document_format(self, filepath: str) -> Dict[str, Any]:
        """Validate document format.

        Args:
            filepath: Path to document

        Returns:
            Validation results
        """
        validation = {
            'exists': os.path.exists(filepath),
            'is_file': os.path.isfile(filepath),
            'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            'extension': os.path.splitext(filepath)[1].lower()
        }
        return validation


    def extract_and_clean_text(self, filepath: str, validation: Dict[str, Any]) -> str:
        """Extract and clean text from document.

        Args:
            filepath: Path to document
            validation: Validation results

        Returns:
            Cleaned text content
        """
        if not validation['exists'] or not validation['is_file']:
            return ""

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.strip()
        except Exception as e:
            logger.warning(f"Error extracting text from {filepath}: {e}")
            return ""


    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for text.

        Args:
            text: Input text

        Returns:
            Dictionary of readability metrics
        """
        if not self.config.enable_readability_metrics:
            return {}

        metrics: Dict[str, float] = {}
        metrics['ari'] = self._calculate_ari(text)
        metrics['cli'] = self._calculate_cli(text)
        metrics['pos_distribution'] = self._calculate_pos_distribution(text)
        return metrics


    def _calculate_pos_distribution(self, text: str) -> float:
        """Calculate part of speech distribution score.

        Args:
            text: Input text

        Returns:
            POS distribution score
        """
        if not self.config.enable_pos_tagging or not text:
            return 0.0

        try:
            # Split text into words
            words = text.split()
            if not words:
                return 0.0

            # Calculate POS tags
            pos_tags: List[Tuple[str, str]] = nltk.pos_tag(words)
            unique_pos: int = len(set(tag for _, tag in pos_tags))
            return unique_pos / len(pos_tags) if pos_tags else 0.0
        except Exception as e:
            logger.warning(f"Error calculating POS distribution: {e}")
            return 0.0


    def _calculate_ari(self, text: str) -> float:
        """Calculate Automated Readability Index.

        Args:
            text: Input text

        Returns:
            ARI score
        """
        if not text or len(text.strip()) <= 1:
            return 0.0

        try:
            # Split text into words and sentences
            words = text.split()
            if not words:
                return 0.0

            # Count characters (excluding whitespace)
            chars = len(re.sub(r'\s+', '', text))

            # Count sentences
            sentences = [s for s in text.split('.') if s.strip()]
            if not sentences:
                return 0.0

            # Calculate ARI
            word_count = len(words)
            sentence_count = len(sentences)
            if word_count < 2 or sentence_count < 1:
                return 0.0

            return 4.71 * (chars / word_count) + 0.5 * (word_count / sentence_count) - 21.43
        except Exception as e:
            logger.warning(f"Error calculating ARI: {e}")
            return 0.0


    def _calculate_cli(self, text: str) -> float:
        """Calculate Coleman-Liau Index.

        Args:
            text: Input text

        Returns:
            CLI score
        """
        if not text or len(text.strip()) <= 1:
            return 0.0

        try:
            # Split text into words and sentences
            words = text.split()
            if not words:
                return 0.0

            # Count characters (excluding whitespace)
            chars = len(re.sub(r'\s+', '', text))

            # Count sentences
            sentences = [s for s in text.split('.') if s.strip()]
            if not sentences:
                return 0.0

            # Calculate CLI
            word_count = len(words)
            sentence_count = len(sentences)
            if word_count < 2 or sentence_count < 1:
                return 0.0

            chars_per_100 = (chars / word_count) * 100
            sentences_per_100 = (sentence_count / word_count) * 100
            return 0.0588 * chars_per_100 - 0.296 * sentences_per_100 - 15.8
        except Exception as e:
            logger.warning(f"Error calculating CLI: {e}")
            return 0.0


    def _get_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of semantic features
        """
        features: Dict[str, Any] = {}
        try:
            if self.config.enable_pos_tagging:
                # Split text into words
                words = text.split()
                if words:
                    try:
                        # Download required NLTK resources if not already downloaded
                        try:
                            nltk.data.find('taggers/averaged_perceptron_tagger')
                        except LookupError:
                            nltk.download('averaged_perceptron_tagger')

                        pos_tags: List[Tuple[str, str]] = nltk.pos_tag(words)
                        pos_counts: Dict[str, int] = {}
                        for _, tag in pos_tags:
                            pos_counts[tag] = pos_counts.get(tag, 0) + 1
                        features["pos_distribution"] = pos_counts
                    except Exception as e:
                        logger.warning(f"Error in POS tagging: {str(e)}")
                        return {}

            if self.config.enable_readability_metrics and "pos_distribution" in features:
                readability_metrics: Dict[str, float] = {
                    "automated_readability_index": self._calculate_ari(text),
                    "coleman_liau_index": self._calculate_cli(text)
                }
                features["readability"] = readability_metrics

        except Exception as e:
            logger.warning(f"Error extracting semantic features: {str(e)}")
            return {}

        return features


def create_analyzer(config: Optional[AnalysisConfig] = None) -> TextAnalyzer:
    """Create a new TextAnalyzer instance.

    Args:
        config: Optional analysis configuration

    Returns:
        Configured TextAnalyzer instance
    """
    return TextAnalyzer(config)
