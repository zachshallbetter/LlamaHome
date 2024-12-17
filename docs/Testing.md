# Testing Guide

## Overview

This document outlines the testing infrastructure and practices for the LlamaHome project. Our testing strategy encompasses unit tests, integration tests, performance tests, and system tests.

## Test Categories

### Unit Tests
- Located in `tests/` directory
- Focus on individual components and functions
- Run with `make test-unit`
- Marked with `@pytest.mark` without specific markers

### Integration Tests
- Test interaction between components
- Located in `tests/` with integration test files
- Run with `make test-integration`
- Marked with `@pytest.mark.integration`

### Performance Tests
- Measure system performance and resource usage
- Located in `tests/training/test_performance.py`
- Run with `make test-performance`
- Marked with `@pytest.mark.performance`

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
- Test configurations are in `.config/test_config.yaml`
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

### Test Fixtures
- Common fixtures in `tests/conftest.py`
- Resource management fixtures
- Mock configurations
- Test data setup

### Best Practices
1. Use descriptive test names
2. One assertion per test when possible
3. Use appropriate markers
4. Include both positive and negative test cases
5. Mock external dependencies
6. Clean up resources in fixtures

## Continuous Integration

### GitHub Actions
- Automated test runs on pull requests
- Coverage reports uploaded as artifacts
- Performance test results tracked

### Test Environment
- Python 3.11
- Dependencies managed through poetry
- GPU tests skipped if no GPU available

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
