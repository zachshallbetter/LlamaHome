# LlamaHome Testing Guide

## Overview

LlamaHome implements a comprehensive testing framework covering unit tests, integration tests, performance benchmarks, and specialized long-context evaluation. The testing infrastructure focuses on model setup validation, device compatibility, memory management, and information retrieval capabilities.

## Test Structure

The test suite is organized as follows:

```text
tests/
│   ├── test_text_analyzer.py
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
make test

# Run benchmarks
make benchmark

# Run needle-in-haystack tests
make needle-test

# Clean test cache
make clean
```

### Test Categories

1. **Unit Tests**

   - Model setup and initialization
   - Device detection (CUDA/MPS/CPU)
   - Memory requirement validation
   - Configuration management
   - Environment variable handling

2. **Integration Tests**
   - Core system integration
   - Request handling
   - Data storage integration
   - Training pipeline
   - GUI integration

3. **Performance Benchmarks**
   - Model loading time
   - Inference speed
   - Memory usage
   - Context window efficiency

4. **Needle Tests**
   - Long context retrieval
   - Information accuracy
   - Context window optimization

## Test Fixtures

### Core Fixtures (conftest.py)

- `test_data_dir`: Temporary test data directory
- `test_config_dir`: Configuration directory
- `test_models_dir`: Model storage directory
- `base_model_config`: Default model configurations
- `clean_env`: Clean environment management

### Model Setup Fixtures

- `mock_config`: Test model configurations
- `setup_test_env`: Test environment with temporary directories

## Configuration

### Model Testing Config

```yaml
models:
  llama3.3:
    name: "Llama 3.3"
    requires_gpu: true
    min_gpu_memory:
      7b: 12
      13b: 24
      70b: 100
    h2o_config:
      enable: true
      window_length: 1024
```

## Contributing Tests

When adding new features:

1. Add corresponding unit tests
2. Include integration tests if component interfaces change
3. Update benchmark tests for performance-critical changes
4. Add needle tests for context window changes
5. Verify against all supported platforms (CPU, CUDA, MPS)

### Test Guidelines

1. Use appropriate fixtures from conftest.py
2. Mock external dependencies
3. Test both success and failure cases
4. Include performance impact tests
5. Document test requirements and setup

### Code Style

All tests should be formatted using:

```bash
make format
make fix
```

## Continuous Integration

Tests are automatically run on:

- Pull requests
- Main branch commits
- Release tags

### CI Pipeline

1. Code style checks
2. Unit tests
3. Integration tests
4. Performance benchmarks
5. Platform-specific tests

## Test Data

Test data locations:

- `tests/data/benchmark_data.jsonl.gz`
- `tests/data/needle_test_data.jsonl.gz`
- `tests/data/test_documents/`

## Environment Requirements

- Python 3.11 or 3.12 (3.13 not supported)
- PyTorch with CUDA support (where applicable)
- Sufficient GPU memory for model tests
