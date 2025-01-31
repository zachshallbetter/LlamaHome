# Testing Guide

## Table of Contents

- [Overview](#overview)
- [Test Categories](#test-categories)
- [Test Data](#test-data)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)
- [Performance Testing](#performance-testing)
- [Debugging Tests](#debugging-tests)
- [Contributing Tests](#contributing-tests)
- [Test Maintenance](#test-maintenance)

## Overview

This document outlines the testing infrastructure and practices for the LlamaHome project. Our testing strategy encompasses unit tests, integration tests, performance tests, and system tests.

## Test Configuration

Testing configuration is managed through `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "security: marks tests as security tests",
]

[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
```

## Test Categories

### Unit Tests

- Located in `tests/` directory
- Focus on individual components
- Run with `make test-unit`

### Integration Tests

- Test component interactions
- Located in `tests/integration/`
- Run with `make test-integration`

### Performance Tests

- Measure system performance
- Located in `tests/performance/`
- Run with `make test-performance`

### System Tests

- End-to-end testing of complete workflows
- Included in integration tests
- Test complete training pipelines

## Test Data

### Sample Data

Test data is located in `data/training/samples/`:

- `test_samples.jsonl`: General test cases
- `performance_samples.jsonl`: Performance testing samples

### Test Configuration

- Test configurations are in `.config/test_config.toml`
- Environment variables for testing in `.env.test`

## Running Tests

### Basic Test Commands

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Run with coverage report
make test-coverage
```

### Test Coverage

- Coverage reports generated in HTML and XML formats
- View HTML report in `htmlcov/index.html`
- Coverage requirements: minimum 80% coverage

## Writing Tests

### Test Structure

```python
import pytest

@pytest.mark.integration  # For integration tests
@pytest.mark.performance  # For performance tests
def test_feature():
    # Test implementation
    pass
```

### Test Dependencies

Development dependencies are managed in `pyproject.toml`:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-qt>=4.2.0",
    "pytest-mock>=3.11.1",
    "pytest-timeout>=2.1.0",
    "pytest-xdist>=3.3.1",
]
```

### Best Practices

1. Use descriptive test names
2. One assertion per test when possible
3. Use appropriate markers
4. Include both positive and negative test cases
5. Mock external dependencies
6. Clean up resources in fixtures

## Test Environment

### Setup

```bash
# Install test dependencies
make setup-test

# Verify test environment
make test-env
```

### Environment Variables

```bash
# Set test environment
export TESTING=true
export PYTEST_ADDOPTS="-v --strict-markers"
```

## Continuous Integration

### GitHub Actions

Tests are automatically run on:
- Pull requests
- Push to main branch
- Scheduled daily runs

### Test Coverage

Coverage requirements:
- Minimum 80% coverage
- Reports generated in HTML and XML
- Coverage tracked in CI/CD

## Performance Testing

### Metrics Tracked

- Execution time
- Memory usage
- GPU utilization
- Disk I/O
- Network bandwidth

### Performance Baselines

- Documented in `docs/Performance.md`
- Updated quarterly
- Regression alerts if performance drops

## Debugging Tests

### Common Issues

1. Resource cleanup in fixtures
2. GPU memory management
3. Test data availability
4. Environment configuration

### Debugging Tools

- pytest-xdist for parallel testing
- pytest-timeout for hanging tests
- pytest-cov for coverage analysis

## Contributing Tests

### Guidelines

1. Follow existing test patterns
2. Include documentation
3. Add appropriate markers
4. Update test data if needed
5. Verify coverage

### Review Process

1. Run full test suite
2. Check coverage impact
3. Verify performance impact
4. Update documentation

## Test Maintenance

### Regular Tasks

1. Update test data
2. Review coverage reports
3. Update performance baselines
4. Clean up deprecated tests

### Quarterly Review

1. Coverage analysis
2. Performance trend analysis
3. Test suite optimization
4. Documentation updates
