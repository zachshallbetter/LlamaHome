# LlamaHome Setup Documentation

LlamaHome is an AI-powered home automation system that integrates the Llama 3.3 model. This document covers the setup process, requirements, and system architecture.

## Prerequisites

- Python 3.11 or 3.12 (3.13 not supported)
- Poetry for dependency management
- Git
- For GPU support:
  - NVIDIA CUDA toolkit 12.1+
  - NVIDIA drivers 525+
  - Minimum GPU memory:
    - 8GB for 7B parameter models
    - 16GB for 13B parameter models
    - 80GB for 70B parameter models

## System Architecture

The setup system consists of several core components that work together:

### Core Components

1. **Makefile** - The entry point that coordinates the setup process:
   - Runs setup commands (`make setup`, `make run`, `make test`, etc.)
   - Manages Poetry environment and dependencies
   - Handles OS/architecture detection
   - Controls code quality and formatting

2. **setup.py** - The main setup orchestrator:
   - Validates system requirements
   - Configures environment and directories
   - Verifies GPU/CUDA compatibility
   - Manages dependency installation

3. **utils/setup_model.py** - Handles AI model management:
   - Configures and downloads models
   - Manages model registry and versioning
   - Detects compute devices (CUDA/MPS/CPU)
   - Provides model setup interface

4. **utils/setup_env.py** - Manages environment configuration:
   - Creates directories and config files
   - Sets up environment variables
   - Configures logging system
   - Provides system information

5. **utils/setup_python.py** - Handles Python environment:
   - Detects compatible Python versions
   - Configures Poetry virtual environment
   - Ensures correct Python version

### Component Interaction Flow

1. `make setup` - Runs setup.py and installs dependencies
2. `make model` - Runs setup_model.py and downloads the Llama 3.3 model
3. `make run` - Runs the main application
