# LlamaHome Style Guide

## Python Code Style

### General Guidelines

- Follow PEP 8 with a max line length of 100 characters
- Use type hints for all function parameters and returns
- Include docstrings for all public functions and classes
- Implement comprehensive error handling
- Write unit tests for all new functionality

### Code Organization

```python
"""Module docstring explaining purpose and functionality."""

import standard_lib
import third_party
from local_module import local_import

class ClassName:
    """Class docstring with full description."""
    
    def __init__(self, param: str) -> None:
        """Initialize with clear parameter documentation."""
        self.param = param
    
    async def method_name(self, param: str) -> str:
        """Method docstring with Args/Returns/Raises."""
        try:
            result = await self._process(param)
            return result
        except Exception as e:
            logger.error(f"Error in method_name: {e}")
            raise
```

### Error Handling

- Use specific exception types
- Include context in error messages
- Log errors appropriately
- Clean up resources in finally blocks

### Testing

- Write unit tests for all functionality
- Include integration tests for component interaction
- Test error cases and edge conditions
- Maintain high test coverage

## Documentation Style

### Markdown Guidelines

- Use ATX headers (#) for sections
- Include code blocks with language specifiers
- Keep line length reasonable (soft limit of 100 characters)
- Use lists for multiple related items

### API Documentation

- Document all endpoints
- Include request/response examples
- List all possible error responses
- Provide usage examples

## Git Workflow

### Commit Messages

```text
type(scope): concise description

Detailed explanation if needed
```

Types:

- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance
