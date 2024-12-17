# LlamaHome

LlamaHome is a powerful document processing and AI integration system, designed for seamless integration with Llama models. It features advanced text preprocessing, semantic analysis, and robust testing infrastructure.

## Quick Setup

### 1. Prerequisites

   ```bash
   # Requires Python 3.11 or 3.12 (3.13 not yet supported)
   python --version
   
   # Install Poetry if not installed
   curl -sSL https://install.python-poetry.org | python3 -
   ```

### 2. One-Step Setup

   ```bash
   git clone https://github.com/zachshallbetter/llamahome.git
   cd llamahome
   make setup
   ```

### 3. Run

   ```bash
   make run
   ```

## Features

### Core Capabilities

- **Advanced Text Processing**
  - Semantic feature extraction
  - POS tagging and distribution analysis
  - Readability metrics (ARI, Coleman-Liau Index)
  - Robust error handling and validation

- **Development Tools**
  - Comprehensive test suite with fixtures
  - Code quality automation (format, fix, check)
  - Integration and unit testing support
  - Performance benchmarking

- **Model Integration**
  - Flexible model configuration
  - Environment-aware setup
  - Cross-platform compatibility
  - Automated resource management

### Development Commands

```bash
# Format and fix code
make fix

# Run tests
make test

# Run benchmarks
make benchmark
make needle-test

# Clean project
make clean
```

## Project Structure

```text
llamahome/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Model integration
│   └── utils/          # Utility functions
├── tests/
│   ├── data/           # Data processing tests
│   └── models/         # Model integration tests
├── utils/              # Development utilities
└── docs/              # Documentation
```

## Configuration

The project uses several configuration files:

- `pyproject.toml` - Project dependencies and settings
- `.flake8` - Code style configuration
- `Makefile` - Build and development commands

## Testing

The test suite includes:

- Unit tests for all core functionality
- Integration tests for end-to-end workflows
- Benchmarks for performance monitoring
- Fixtures for consistent test data

Run tests with:

```bash
make test                 # Run all tests
pytest tests/data/        # Run specific test directory
pytest -m integration     # Run integration tests
```

## Documentation

- [Features](docs/Features.md) - Detailed feature documentation
- [Architecture](docs/Architecture.md) - System design
- [Testing](docs/Testing.md) - Testing guide
- [Development](docs/Development.md) - Development guide

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run checks (`make fix && make check`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE).
