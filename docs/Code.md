# LlamaHome Python Code Style Guide

This guide outlines the coding standards and best practices for the LlamaHome project.

## General Principles

### Readability

- Write self-explanatory code with meaningful names
- Keep functions and methods focused and concise
- Use clear logic and avoid complex nesting
- Add comments only when necessary to explain "why", not "what"

### Consistency

- Follow PEP 8 for Python code style
- Maintain consistent naming conventions
- Use consistent indentation (4 spaces)
- Follow consistent file structure

### Modernity

- Use Python 3.10+ features
- Leverage standard library capabilities
- Use well-maintained external libraries
- Avoid deprecated APIs and patterns

## Code Organization

### File Structure

```text
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── model.py
│   └── config.py
├── api/
│   ├── __init__.py
│   └── routes.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

### Imports

Group imports in this order:

```python
# 1. Standard library imports
import os
import sys
from typing import Optional

# 2. Third-party imports
import numpy as np
import torch

# 3. Local imports
from .core import model
from .utils import helpers
```

### Naming Conventions

#### Classes

```python
class ModelManager:
    """Manages Llama model instances."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self._models = {}
```

#### Functions

```python
def process_prompt(
    text: str,
    temperature: float = 0.7
) -> str:
    """Process user prompt with the model.
    
    Args:
        text: The input prompt text
        temperature: Sampling temperature
        
    Returns:
        Generated response text
    """
    return model.generate(text, temperature)
```

#### Variables and Constants

```python
# Module-level constants
MAX_SEQUENCE_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.7

# Local variables
model_name = "llama3.3"
response_text = process_prompt("Hello")
```

## Testing

### Unit Tests

```python
def test_process_prompt():
    """Test prompt processing functionality."""
    result = process_prompt("Test input")
    assert isinstance(result, str)
    assert len(result) > 0
```

### Integration Tests

```python
class TestModelIntegration:
    """Test model integration with API."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_model_response(self):
        response = self.client.post(
            "/api/process_prompt",
            json={"prompt": "Test"}
        )
        assert response.status_code == 200
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
```

### Error Handling Pattern

```python
try:
    result = process_prompt(user_input)
except ModelError as e:
    logger.error(f"Model error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
except ConfigError as e:
    logger.error(f"Config error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
```

## Documentation

### Docstring Format

```python
def validate_config(config: dict) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ConfigError: If configuration is invalid
        
    Example:
        >>> config = {"model_path": "/path/to/model"}
        >>> validate_config(config)
        True
    """
    pass
```

### Type Hints

```python
from typing import Dict, List, Optional, Union

def get_model_info(
    model_name: str,
    include_stats: bool = False
) -> Dict[str, Union[str, int, float]]:
    """Get model information."""
    pass
```

## Performance

### Optimization Tips

- Use generators for large datasets
- Implement caching where appropriate
- Profile code regularly
- Use async/await for I/O operations
- Batch operations when possible

### Memory Management

- Release resources explicitly
- Use context managers
- Implement proper cleanup
- Monitor memory usage
- Handle large files in chunks

## Version Control

### Commit Messages

```text
feat: add new model configuration option
^--^  ^-----------------------------^
|     |
|     +-> Summary in present tense
|
+-------> Type: feat, fix, docs, style, refactor, test, chore
```

### Branch Naming

```text
feature/add-model-config
bugfix/memory-leak
docs/update-api-docs
```

## Security

- Never commit sensitive data
- Use environment variables
- Implement proper authentication
- Validate all inputs
- Follow security best practices
