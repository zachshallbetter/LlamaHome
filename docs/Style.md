# LlamaHome Style Guide

## Code Style Overview

### Core Principles

1. **Clarity First**
   - Write self-documenting code
   - Use descriptive names
   - Keep functions focused
   - Maintain consistent style

2. **Code Organization**

   ```text
   src/
   ├── core/           # Core functionality
   ├── models/         # Model implementations
   ├── utils/          # Utility functions
   ├── config/         # Configuration handling
   ├── api/            # API implementations
   └── interfaces/     # User interfaces
   ```

## Python Style Guide

### Code Formatting

1. **Basic Formatting**

   ```python
   # Good
   def process_request(
       self,
       request: Request,
       timeout: int = 30
   ) -> Response:
       """Process incoming request."""
       return self.handler.handle(request, timeout)
   
   # Bad
   def process_request(self, request: Request, timeout: int = 30) -> Response:
       """Process incoming request."""
       return self.handler.handle(request, timeout)
   ```

2. **Import Formatting**

   ```python
   # Standard library imports
   import os
   import sys
   from typing import Dict, List, Optional
   
   # Third-party imports
   import numpy as np
   import torch
   from fastapi import FastAPI
   
   # Local imports
   from llamahome.core import CoreSystem
   from llamahome.utils import logger
   ```

### Naming Conventions

1. **Variable Names**

   ```python
   # Good
   model_config = ModelConfig()
   user_response = get_user_input()
   is_valid = validate_input(user_response)
   
   # Bad
   conf = ModelConfig()
   resp = get_user_input()
   valid = validate_input(resp)
   ```

2. **Function Names**

   ```python
   # Good
   def initialize_model():
       """Initialize model with configuration."""
       pass
   
   def process_user_request():
       """Process user request with validation."""
       pass
   
   # Bad
   def init_mod():
       pass
   
   def proc_req():
       pass
   ```

### Type Hints

1. **Basic Types**

   ```python
   def process_data(
       data: List[str],
       batch_size: int,
       temperature: float = 0.7
   ) -> Dict[str, Any]:
       """Process data in batches."""
       results = {}
       for batch in chunks(data, batch_size):
           results.update(process_batch(batch, temperature))
       return results
   ```

2. **Complex Types**

   ```python
   from typing import TypeVar, Generic, Protocol
   
   T = TypeVar('T')
   
   class DataProcessor(Generic[T]):
       """Generic data processor."""
       
       def process(self, data: T) -> T:
           """Process data of type T."""
           return data
   ```

### Documentation

1. **Function Documentation**

   ```python
   def train_model(
       model: Model,
       dataset: Dataset,
       epochs: int = 10,
       learning_rate: float = 0.001
   ) -> TrainingResults:
       """
       Train model on dataset.
       
       Args:
           model: Model instance to train
           dataset: Training dataset
           epochs: Number of training epochs
           learning_rate: Learning rate for optimization
           
       Returns:
           TrainingResults containing metrics and model state
           
       Raises:
           ValueError: If dataset is empty
           ResourceError: If insufficient GPU memory
       """
       pass
   ```

2. **Class Documentation**

   ```python
   class ModelManager:
       """
       Manage model lifecycle and resources.
       
       This class handles model initialization, loading,
       unloading, and resource management. It ensures
       proper cleanup and optimal resource utilization.
       
       Attributes:
           models: Dictionary of loaded models
           active_model: Currently active model
           config: Model configuration
           
       Example:
           manager = ModelManager()
           model = await manager.load_model("llama3.3")
           result = await model.process(prompt)
       """
       pass
   ```

### Error Handling

1. **Exception Hierarchy**

   ```python
   class LlamaHomeError(Exception):
       """Base exception for LlamaHome."""
       pass
   
   class ModelError(LlamaHomeError):
       """Model-related errors."""
       pass
   
   class ConfigError(LlamaHomeError):
       """Configuration-related errors."""
       pass
   ```

2. **Error Handling Patterns**

   ```python
   async def process_safely(
       self,
       request: Request
   ) -> Response:
       """Process request with proper error handling."""
       try:
           return await self._process(request)
       except ModelError as e:
           logger.error(f"Model error: {e}")
           return ErrorResponse(str(e))
       except ConfigError as e:
           logger.error(f"Config error: {e}")
           return ErrorResponse(str(e))
       except Exception as e:
           logger.exception("Unexpected error")
           return ErrorResponse("Internal error")
   ```

## Testing Style

### Test Organization

1. **Test Structure**

   ```python
   @pytest.mark.core
   class TestModelManager:
       """Test model management functionality."""
       
       @pytest.fixture
       def manager(self):
           """Provide test manager instance."""
           return ModelManager()
       
       def test_initialization(self, manager):
           """Test manager initialization."""
           assert manager.is_initialized
           
       @pytest.mark.asyncio
       async def test_model_loading(self, manager):
           """Test model loading process."""
           model = await manager.load_model("llama3.3")
           assert model.is_loaded
   ```

2. **Test Naming**

   ```python
   def test_should_process_valid_request():
       """Test processing of valid request."""
       pass
   
   def test_should_reject_invalid_request():
       """Test rejection of invalid request."""
       pass
   
   def test_should_handle_timeout_gracefully():
       """Test graceful timeout handling."""
       pass
   ```

### Test Documentation

1. **Test Case Documentation**

   ```python
   def test_model_initialization():
       """
       Test model initialization process.
       
       Scenario:
           1. Create model instance
           2. Initialize with configuration
           3. Verify initialization state
           
       Expected:
           - Model should be properly initialized
           - Resources should be allocated
           - Configuration should be applied
       """
       pass
   ```

2. **Test Suite Documentation**

   ```python
   @pytest.mark.integration
   class TestAPIIntegration:
       """
       Test API integration functionality.
       
       This test suite verifies the integration between
       the API layer and the core system. It ensures
       proper request handling, error management, and
       resource cleanup.
       
       Requirements:
           - Running API server
           - Test database
           - Mock authentication
       """
       pass
   ```

## Configuration Style

### YAML Configuration

1. **System Configuration**

   ```yaml
   # system_config.toml
   system:
     log_level: INFO
     cache_size: 10GB
     max_memory: 90%
   
   performance:
     batch_size: 32
     num_workers: 4
     timeout: 30
   
   security:
     enable_auth: true
     token_expiry: 3600
     max_requests: 1000
   ```

2. **Model Configuration**

   ```yaml
   # model_config.toml
   models:
     llama3.3:
       version: "3.3"
       variants:
         - "7b"
         - "13b"
         - "70b"
       context_length: 32768
       optimization:
         quantization: "int8"
         gpu_layers: 32
   ```

### Environment Configuration

1. **Environment Variables**

   ```bash
   # .env
   LLAMAHOME_ENV=development
   LLAMAHOME_LOG_LEVEL=INFO
   LLAMAHOME_CACHE_DIR=/path/to/cache
   LLAMAHOME_MODEL_DIR=/path/to/models
   ```

2. **Environment Validation**

   ```python
   def validate_environment():
       """
       Validate environment configuration.
       
       Required variables:
           - LLAMAHOME_ENV
           - LLAMAHOME_LOG_LEVEL
           - LLAMAHOME_CACHE_DIR
           - LLAMAHOME_MODEL_DIR
       """
       required_vars = [
           "LLAMAHOME_ENV",
           "LLAMAHOME_LOG_LEVEL",
           "LLAMAHOME_CACHE_DIR",
           "LLAMAHOME_MODEL_DIR"
       ]
       
       for var in required_vars:
           if var not in os.environ:
               raise ConfigError(f"Missing {var}")
   ```

## API Style

### REST API

1. **Endpoint Structure**

   ```python
   @app.post("/api/v1/process")
   async def process_request(
       request: Request,
       current_user: User = Depends(get_current_user)
   ) -> Response:
       """
       Process user request.
       
       Args:
           request: User request
           current_user: Authenticated user
           
       Returns:
           Response containing processed result
       """
       return await process_handler.handle(request)
   ```

2. **Response Format**

   ```python
   class APIResponse(BaseModel):
       """API response format."""
       
       status: str
       message: str
       data: Optional[Dict[str, Any]]
       error: Optional[str]
       
       class Config:
           """Response configuration."""
           schema_extra = {
               "example": {
                   "status": "success",
                   "message": "Request processed",
                   "data": {"result": "..."},
                   "error": None
               }
           }
   ```

## Documentation Style

### Markdown Documentation

1. **File Structure**

   ```markdown
   # Component Title
   
   Brief description of the component.
   
   ## Overview
   
   Detailed component description.
   
   ## Usage
   
   Usage examples and patterns.
   
   ## API Reference
   
   Detailed API documentation.
   
   ## Examples
   
   Code examples and use cases.
   ```

2. **Code Examples**

   ```markdown
   ### Basic Usage
   

   ```python
   from llamahome import LlamaHome
   
   app = LlamaHome()
   result = await app.process("Hello")
   print(result)
   ```

### Advanced Configuration

   ```python
   app = LlamaHome(
       model="llama3.3",
       config={
           "temperature": 0.7,
           "max_tokens": 100
       }
   )
   ```

# Markdown Style Guide

## Overview

This guide defines the Markdown formatting standards for LlamaHome documentation. These rules are enforced by markdownlint and ensure consistency across all documentation.

## Basic Rules

### Headings

- Use ATX-style headings with `#` symbols
- Include one space after the `#`
- Leave one blank line before and after headings
- Increment heading levels by one only (no skipping levels)

```markdown
# Top Level Heading

## Second Level

### Third Level
```

### Lists

- Use `-` for unordered lists
- Use `1.` for ordered lists (can also use sequential numbers)
- Indent nested lists with 2 spaces
- Leave one blank line before and after lists

```text
- First item
  - Nested item
  - Another nested item
- Second item

1. First ordered item
2. Second ordered item
   1. Nested ordered item
   2. Another nested item
```

### Code Blocks

- Use fenced code blocks with backticks (```)
- Always specify the language
- Leave one blank line before and after code blocks

```python
def example():
    return "This is a code block"
```

### Emphasis and Strong Emphasis

- Use single asterisks for *emphasis*
- Use double asterisks for **strong emphasis**
- No spaces inside emphasis markers

```markdown
This is *emphasized* text
This is **strong** text
```

## Spacing and Length

### Line Length

- Maximum line length is 120 characters
- Exceptions:
  - Code blocks
  - Tables
  - Headings
  - URLs

### Blank Lines

- Maximum of 2 consecutive blank lines
- One blank line before and after:
  - Headings
  - Lists
  - Code blocks
  - Blockquotes

## Links and URLs

- Use reference-style links for repeated URLs
- Bare URLs are allowed in some cases
- No spaces inside link text
- Include alt text for images

```markdown
[Link text][reference]
[Direct link](https://example.com)
![Alt text](image.png "Optional title")

[reference]: https://example.com
```

## HTML Elements

Allowed HTML elements:

- `<br>` - Line breaks
- `<details>` - Collapsible sections
- `<summary>` - Details summary
- `<kbd>` - Keyboard input
- `<sup>` - Superscript
- `<sub>` - Subscript

## Proper Names

The following names must be capitalized correctly:

### Project Names

- LlamaHome
- H2O
- TensorBoard
- PyTorch
- Hugging Face
- PEFT
- LoRA

### Languages and Frameworks

- Python
- JavaScript
- TypeScript
- React
- Node.js
- FastAPI
- Django

### Tools and Platforms

- GitHub
- GitLab
- Docker
- Kubernetes
- VS Code
- PyCharm
- Jupyter

### Technologies

- CUDA
- GPU
- CPU
- YAML
- JSON
- REST
- API
- ML
- AI
- LLM

### Operating Systems

- Linux
- macOS
- Windows
- Ubuntu
- CentOS

## File Structure

### Required Elements

- Every file should have a title (H1)
- Include a brief description after the title
- End file with a single newline

### Optional Elements

- Table of contents for longer documents
- Next steps or related documents section
- Code examples when relevant

## Examples

### Document Template

```markdown
# Document Title

Brief description of the document's purpose.

## Table of Contents

- [Section One](#section-one)
- [Section Two](#section-two)

## Section One

Content for section one.

## Section Two

Content for section two.

## Next Steps

1. [Related Document One](doc1.md)
2. [Related Document Two](doc2.md)
```

### Code Documentation

```markdown
## Function Description

```python
def process_data(input: str) -> Dict[str, Any]:
    """Process input data."""
    return {"result": input}
```

Key features:

- Input validation
- Error handling
- Return type specification

## Common Issues

### Avoid

- Mixing emphasis styles (*asterisk* and *underscore*)
- Skipping heading levels (# then ###)
- Empty links or images
- Inconsistent list marker spacing
- Hard tabs (use spaces)

### Prefer

- Consistent emphasis style (asterisks)
- Sequential heading levels
- Complete link references
- Proper list indentation
- Soft wrapping for long lines

## Tools and Enforcement

### markdownlint

Our Markdown style is enforced using markdownlint with custom configuration:

```json
{
  "default": true,
  "MD013": {
    "line_length": 120,
    "code_blocks": false,
    "tables": false
  }
}
```

### VS Code Settings

Recommended VS Code settings for Markdown:

```json
{
  "editor.wordWrap": "on",
  "editor.rulers": [120],
  "markdown.preview.breaks": true
}
```

## Next Steps

1. [Code Style Guide](Code.md)
2. [Documentation Guide](Documentation.md)
3. [Contributing Guide](Contributing.md)
