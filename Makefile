# Environment and Python settings
PYTHON := python3.11
VENV := .venv
BIN := $(VENV)/bin
PYTHON_VERSION := $(shell $(PYTHON) --version | cut -d' ' -f2)

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# PyTorch installation command
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        TORCH_INSTALL := pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    else
        TORCH_INSTALL := pip3 install torch torchvision torchaudio
    endif
else
    TORCH_INSTALL := pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
endif

# Directories
CACHE_DIR := .cache
DATA_DIR := .data
CONFIG_DIR := .config
LOGS_DIR := .logs

# Cache subdirectories
MODEL_CACHE := $(CACHE_DIR)/models
TRAINING_CACHE := $(CACHE_DIR)/training
SYSTEM_CACHE := $(CACHE_DIR)/system
PYCACHE := $(CACHE_DIR)/pycache

# Data subdirectories
TRAINING_DATA := $(DATA_DIR)/training
MODEL_DATA := $(DATA_DIR)/models
METRICS_DATA := $(DATA_DIR)/metrics

# Config files
TRAINING_CONFIG := $(CONFIG_DIR)/training_config.toml
MODELS_CONFIG := $(CONFIG_DIR)/models.json
ENV_FILE := .env

# Platform detection
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
    VENV_PYTHON := $(VENV_BIN)/python.exe
    VENV_PIP := $(VENV_BIN)/pip.exe
    VENV_ACTIVATE := $(VENV_BIN)/activate.bat
    RM := rmdir /s /q
else
    VENV_BIN := $(VENV)/bin
    VENV_PYTHON := $(VENV_BIN)/python
    VENV_PIP := $(VENV_BIN)/pip
    VENV_ACTIVATE := $(VENV_BIN)/activate
    RM := rm -rf
endif

# Data directories
MODELS_DIR := $(DATA_DIR)/models
DATASETS_DIR := $(DATA_DIR)/datasets
EMBEDDINGS_DIR := $(DATA_DIR)/embeddings
CHECKPOINTS_DIR := $(DATA_DIR)/checkpoints
ARTIFACTS_DIR := $(DATA_DIR)/artifacts
MEMORY_DIR := $(DATA_DIR)/memory
METRICS_DIR := $(DATA_DIR)/metrics
TELEMETRY_DIR := $(DATA_DIR)/telemetry
TRAINING_DIR := $(DATA_DIR)/training
LOCAL_DIR := $(DATA_DIR)/local

# Create data directories
.PHONY: setup-data
setup-data:
	mkdir -p $(MODELS_DIR) \
		$(DATASETS_DIR) \
		$(EMBEDDINGS_DIR) \
		$(CHECKPOINTS_DIR) \
		$(ARTIFACTS_DIR) \
		$(MEMORY_DIR) \
		$(METRICS_DIR) \
		$(TELEMETRY_DIR) \
		$(TRAINING_DIR) \
		$(LOCAL_DIR)

# Default target
.PHONY: all
all: setup

# Directory setup
.PHONY: setup-dirs
setup-dirs:
	@echo "Creating directory structure..."
	@mkdir -p $(MODEL_CACHE) $(TRAINING_CACHE) $(SYSTEM_CACHE) $(PYCACHE)
	@mkdir -p $(TRAINING_DATA) $(MODEL_DATA) $(METRICS_DATA)
	@mkdir -p $(CONFIG_DIR) $(LOGS_DIR)
	@echo "Directory structure created"

# Environment setup
.PHONY: setup-env
setup-env:
	@echo "Setting up Python environment..."
	@if [ -d "$(VENV)" ]; then \
		echo "Virtual environment already exists. Cleaning..."; \
		$(RM) $(VENV); \
	fi
	@$(PYTHON) -m venv $(VENV)
	@echo "Activating virtual environment..."
	@. $(VENV_ACTIVATE) && \
	echo "Upgrading pip..." && \
	$(VENV_PIP) install --upgrade pip && \
	echo "Installing PyTorch..." && \
	$(TORCH_INSTALL) && \
	echo "Installing project and dependencies..." && \
	$(VENV_PIP) install -e ".[dev]" || \
	(echo "Failed to install dependencies. Check pyproject.toml" && exit 1)

# Configuration setup
.PHONY: setup-config
setup-config:
	@echo "Setting up configuration..."
	@test -f $(TRAINING_CONFIG) || cp config/training_config.toml.example $(TRAINING_CONFIG)
	@test -f $(MODELS_CONFIG) || cp config/models.json.example $(MODELS_CONFIG)
	@test -f $(ENV_FILE) || cp .env.example $(ENV_FILE)
	@echo "Configuration files ready"

# Main setup
.PHONY: setup
setup: setup-dirs setup-env setup-config
	@echo "Setup complete"

# Development setup
.PHONY: setup-dev
setup-dev: setup
	@echo "Installing development dependencies..."
	@. $(VENV_ACTIVATE) && $(VENV_PIP) install -e ".[dev]" || \
	(echo "Failed to install development dependencies" && exit 1)

# Test setup
.PHONY: setup-test
setup-test: setup
	@echo "Installing test dependencies..."
	@. $(VENV_ACTIVATE) && $(VENV_PIP) install -e ".[test]" || \
	(echo "Failed to install test dependencies" && exit 1)

# Clean targets
.PHONY: clean-pycache
clean-pycache:
	@echo "Cleaning Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@$(RM) $(PYCACHE)
	@echo "Python cache cleaned"

.PHONY: clean-cache
clean-cache:
	@echo "Cleaning cache directories..."
	@$(RM) $(MODEL_CACHE) $(TRAINING_CACHE) $(SYSTEM_CACHE)
	@echo "Cache directories cleaned"

.PHONY: clean-all
clean-all: clean-pycache clean-cache
	@echo "Cleaning all temporary files..."
	@$(RM) $(VENV) .pytest_cache .mypy_cache .ruff_cache
	@echo "All temporary files cleaned"

# Code quality
.PHONY: lint
lint:
	@echo "Running linters..."
	@. $(VENV_ACTIVATE) && ruff check .
	@. $(VENV_ACTIVATE) && mypy .
	@. $(VENV_ACTIVATE) && black --check .
	@. $(VENV_ACTIVATE) && bandit -r src/

.PHONY: format
format:
	@echo "Formatting code..."
	@. $(VENV_ACTIVATE) && black .
	@. $(VENV_ACTIVATE) && ruff check --fix .

# Training targets
.PHONY: train
train:
	@echo "Starting training..."
	@. $(VENV_ACTIVATE) && $(VENV_PYTHON) -m src.interfaces.cli train $(filter-out $@,$(MAKECMDGOALS))

.PHONY: train-resume
train-resume:
	@echo "Resuming training..."
	@. $(VENV_ACTIVATE) && $(VENV_PYTHON) -m src.interfaces.cli train-resume $(filter-out $@,$(MAKECMDGOALS))

.PHONY: train-eval
train-eval:
	@echo "Evaluating model..."
	@. $(VENV_ACTIVATE) && $(VENV_PYTHON) -m src.interfaces.cli train-eval $(filter-out $@,$(MAKECMDGOALS))

# Distributed Training
.PHONY: train-distributed train-multi-node

train-distributed:
	@echo "Starting distributed training..."
	@. $(VENV_ACTIVATE) && $(VENV_PYTHON) -m src.training.launch \
		--config .config/distributed_config.yaml \
		--data .data/training \
		--output .data/output/training \
		--epochs $(EPOCHS) \
		--world-size $(WORLD_SIZE)

train-multi-node:
	@echo "Starting multi-node training..."
	@. $(VENV_ACTIVATE) && $(VENV_PYTHON) -m src.training.launch \
		--config .config/distributed_config.yaml \
		--data .data/training \
		--output .data/output/training \
		--epochs $(EPOCHS) \
		--num-nodes $(NUM_NODES) \
		--node-rank $(NODE_RANK) \
		--master-addr $(MASTER_ADDR) \
		--master-port $(MASTER_PORT)

# Default values for training parameters
EPOCHS ?= 10
WORLD_SIZE ?= $(shell python -c "import torch; print(torch.cuda.device_count())")
NUM_NODES ?= 1
NODE_RANK ?= 0
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29500

# Testing targets
.PHONY: test test-unit test-integration test-coverage test-performance test-specialized

test: test-unit test-integration test-specialized

test-unit:
	@echo "Running unit tests..."
	@. $(VENV_ACTIVATE) && pytest tests/ -v -m "not integration and not performance and not specialized"

test-integration:
	@echo "Running integration tests..."
	@. $(VENV_ACTIVATE) && pytest tests/ -v -m "integration"

test-performance:
	@echo "Running performance tests..."
	@. $(VENV_ACTIVATE) && pytest tests/performance/ -v -m "performance"

test-specialized:
	@echo "Running specialized tests..."
	@. $(VENV_ACTIVATE) && pytest tests/specialized/ -v -m "specialized"

test-coverage:
	@echo "Running tests with coverage..."
	coverage run -m pytest
	coverage xml
	coverage html

# Run targets
.PHONY: run
run:
	@echo "Starting LlamaHome..."
	@$(BIN)/python -m src.interfaces.cli

.PHONY: shell
shell:
	@echo "Starting LlamaHome shell..."
	@$(BIN)/python -m src.interfaces.cli shell

# Allow passing arguments to targets
%:
	@:
