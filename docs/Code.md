# LlamaHome Python Code Style Guide

## Table of Contents

- [General Principles](#general-principles)
- [Readability](#readability)
- [Consistency](#consistency)
- [Code Organization](#code-organization)
- [Testing](#testing)
- [Error Handling](#error-handling)
- [Documentation](#documentation)
- [Performance](#performance)
- [Version Control](#version-control)
- [Security](#security)
- [Contributing](#contributing)

## Overview

This document provides a comprehensive overview of LlamaHome's coding style guide, including general principles, readability, consistency, code organization, testing, error handling, documentation, performance, version control, security, and contributing.

## General Principles

### Readability

- Write self-explanatory code with meaningful names
- Keep functions and methods focused and concise
- Use clear logic and avoid complex nesting
- Add comments only when necessary to explain "why", not "what"
- Maintain comprehensive inline documentation for complex algorithms

### Consistency

- Follow PEP 8 for Python code style
- Maintain consistent naming conventions
- Use consistent indentation (4 spaces)
- Follow consistent file structure
- Ensure uniform error handling patterns

### Modernity

- Use Python 3.11 features (3.13 not yet supported)
- Leverage standard library capabilities
- Use well-maintained external libraries (llama-recipes, h2o)
- Avoid deprecated APIs and patterns

## Code Organization

### File Structure

```text
src/
├── core/
│   ├── __init__.py
│   ├── attention.py     # Hybrid attention mechanisms
│   ├── model.py         # Core model implementation
│   ├── cache.py         # Caching system
│   └── config_handler.py # Configuration management
├── training/
│   ├── __init__.py
│   ├── pipeline.py      # Training orchestration
│   ├── data.py         # Data management
│   ├── resources.py    # Resource handling
│   ├── monitoring.py   # Training monitoring
│   └── optimization.py # Training optimization
└── utils/
    ├── __init__.py
    ├── log_manager.py  # Singleton logging
    ├── model_manager.py # Model lifecycle
    └── cache_manager.py # Cache management
```

### Imports

Group imports in this order:

```python
# 1. Standard library imports
import os
import sys
from typing import Optional, Dict, List, Union

# 2. Third-party imports
import torch
from llama_recipes import Trainer
import h2o

# 3. Local imports
from .core.model import Model
from .utils.log_manager import LogManager
```

### Naming Conventions

#### Classes

```python
class ConfigManager:
    """Manages configuration across the system.

    Implements a singleton pattern for centralized config management
    with support for multiple config sources and validation.
    """

    _instance = None

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._load_configs()
```

#### Functions

```python
async def process_model_input(
    text: str,
    config: Dict[str, Any],
    cache_manager: Optional[CacheManager] = None
) -> str:
    """Process input text through the model with caching support.

    Args:
        text: The input text to process
        config: Model configuration dictionary
        cache_manager: Optional cache manager instance

    Returns:
        Processed text output

    Raises:
        ModelError: If model processing fails
        ConfigError: If configuration is invalid
    """
    try:
        if cache_manager and cache_manager.has_cached_result(text):
            return cache_manager.get_cached_result(text)

        result = await model.process(text, config)

        if cache_manager:
            cache_manager.cache_result(text, result)

        return result
    except Exception as e:
        logger.error(f"Error processing model input: {e}")
        raise ModelError(f"Failed to process input: {e}")
```

## Testing

### Unit Tests

```python
class TestConfigManager:
    """Test configuration management functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.config_manager = ConfigManager()
        self.test_config = {
            "model_path": "/path/to/model",
            "cache_enabled": True
        }

    def test_config_validation(self):
        """Test configuration validation logic."""
        result = self.config_manager.validate_config(self.test_config)
        assert result.is_valid
        assert len(result.errors) == 0
```

### Integration Tests

```python
@pytest.mark.integration
class TestModelIntegration:
    """Test model integration with training pipeline."""

    async def test_training_pipeline(self):
        """Test end-to-end training pipeline."""
        config = self.load_test_config()
        pipeline = TrainingPipeline(config)

        result = await pipeline.run_training()

        assert result.status == "success"
        assert result.metrics["loss"] < 0.1
```

## Error Handling

### Exception Hierarchy

```python
class LlamaHomeError(Exception):
    """Base exception for LlamaHome."""
    pass

class ModelError(LlamaHomeError):
    """Model-related errors."""
    pass

class ConfigError(LlamaHomeError):
    """Configuration errors."""
    pass

class CacheError(LlamaHomeError):
    """Cache-related errors."""
    pass
```

### Error Handling Pattern

```python
try:
    config = config_manager.load_config()
    model = await model_manager.initialize_model(config)
    result = await model.process(input_text)
except ConfigError as e:
    logger.error(f"Configuration error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except ModelError as e:
    logger.error(f"Model error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
finally:
    await model_manager.cleanup()
```

## Documentation

### Docstring Format

```python
def train_model(
    config: Dict[str, Any],
    data_path: str,
    cache_enabled: bool = True
) -> TrainingResult:
    """Train model with specified configuration and data.

    Implements hybrid training approach combining llama-recipes
    and H2O features for optimal performance.

    Args:
        config: Training configuration dictionary
        data_path: Path to training data
        cache_enabled: Whether to enable training cache

    Returns:
        TrainingResult object containing metrics and model state

    Raises:
        TrainingError: If training fails
        DataError: If data loading fails

    Example:
        >>> config = load_config("training_config.toml")
        >>> result = train_model(config, "data/training")
        >>> print(f"Training loss: {result.metrics['loss']}")
    """
    pass
```

### Type Hints

```python
from typing import Dict, List, Optional, Union, TypeVar, Generic

T = TypeVar('T')

class CacheManager(Generic[T]):
    """Generic cache manager supporting different value types."""

    def get_cached_value(
        self,
        key: str,
        default: Optional[T] = None
    ) -> Optional[T]:
        """Retrieve cached value."""
        pass
```

## Performance

### Optimization Tips

- Use generators for large datasets
- Implement proper caching strategies
- Profile code regularly
- Use async/await for I/O operations
- Batch operations when possible
- Leverage GPU acceleration when available
- Implement proper memory management
- Use appropriate data structures

### Memory Management

- Release resources explicitly
- Use context managers
- Implement proper cleanup
- Monitor memory usage
- Handle large files in chunks
- Implement cache eviction policies
- Use weak references when appropriate

## Version Control

### Commit Messages

```text
type(scope): concise description

Detailed explanation if needed

Resolves: #123
See also: #456, #789
```

Types:

- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance
- perf: Performance improvement

### Branch Naming

```text
feature/add-hybrid-attention
bugfix/memory-leak-fix
docs/update-architecture
perf/optimize-cache
```

## Security

- Never commit sensitive data
- Use environment variables for secrets
- Implement proper authentication
- Validate all inputs
- Follow security best practices
- Implement proper access controls
- Use secure communication channels
- Regular security audits

### Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
