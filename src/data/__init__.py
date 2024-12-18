"""Data handling components for LlamaHome.

This package provides data processing, storage, training data management and text analysis:
- Storage: Efficient data storage and retrieval
- Training: Training data preparation and management
- Analysis: Text analysis, readability scoring, and linguistic features
- Conversion: Format conversion between different file types
"""


__all__ = [
    # Storage
    'DataStorage',
    'create_storage',

    # Training
    'TrainingData',
    'create_training',

    # Analysis
    'TextAnalyzer',
    'AnalysisConfig',
    'create_analyzer',

    # Conversion
    'FormatConverter',
    'create_converter',

    # Cache
    'DataCache',
    'create_cache',
]
