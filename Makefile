# Environment and Python settings
PYTHON := python3.11
VENV := .venv
BIN := $(VENV)/bin
PYTHON_VERSION := $(shell $(PYTHON) --version | cut -d' ' -f2)

# Directories
CACHE_DIR := .cache
DATA_DIR := data
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
TRAINING_CONFIG := $(CONFIG_DIR)/training_config.yaml
MODELS_CONFIG := $(CONFIG_DIR)/models.json
ENV_FILE := .env

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
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(BIN)/pip install --upgrade pip
	@$(BIN)/pip install poetry
	@$(BIN)/poetry install
	@echo "Python environment ready"

# Configuration setup
.PHONY: setup-config
setup-config:
	@echo "Setting up configuration..."
	@test -f $(TRAINING_CONFIG) || cp config/training_config.yaml.example $(TRAINING_CONFIG)
	@test -f $(MODELS_CONFIG) || cp config/models.json.example $(MODELS_CONFIG)
	@test -f $(ENV_FILE) || cp .env.example $(ENV_FILE)
	@echo "Configuration files ready"

# Main setup
.PHONY: setup
setup: setup-dirs setup-env setup-config
	@echo "Setup complete"

# Clean targets
.PHONY: clean-pycache
clean-pycache:
	@echo "Cleaning Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf $(PYCACHE)/*
	@echo "Python cache cleaned"

.PHONY: clean-cache
clean-cache:
	@echo "Cleaning cache directories..."
	@rm -rf $(MODEL_CACHE)/* $(TRAINING_CACHE)/* $(SYSTEM_CACHE)/*
	@echo "Cache directories cleaned"

.PHONY: clean-all
clean-all: clean-pycache clean-cache
	@echo "Cleaning all temporary files..."
	@rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache
	@echo "All temporary files cleaned"

# Training targets
.PHONY: train
train:
	@echo "Starting training..."
	@$(BIN)/python -m src.interfaces.cli train $(filter-out $@,$(MAKECMDGOALS))

.PHONY: train-resume
train-resume:
	@echo "Resuming training..."
	@$(BIN)/python -m src.interfaces.cli train-resume $(filter-out $@,$(MAKECMDGOALS))

.PHONY: train-eval
train-eval:
	@echo "Evaluating model..."
	@$(BIN)/python -m src.interfaces.cli train-eval $(filter-out $@,$(MAKECMDGOALS))

# Distributed Training
.PHONY: train-distributed train-multi-node

train-distributed:
	@echo "Starting distributed training..."
	python -m src.training.launch \
		--config .config/distributed_config.yaml \
		--data data/training \
		--output output/training \
		--epochs $(EPOCHS) \
		--world-size $(WORLD_SIZE)

train-multi-node:
	@echo "Starting multi-node training..."
	python -m src.training.launch \
		--config .config/distributed_config.yaml \
		--data data/training \
		--output output/training \
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

# Development targets
.PHONY: test test-unit test-integration test-coverage test-performance test-specialized

test: test-unit test-integration test-specialized

test-unit:
	@echo "Running unit tests..."
	@$(BIN)/pytest tests/ -v -m "not integration and not performance and not specialized"

test-integration:
	@echo "Running integration tests..."
	@$(BIN)/pytest tests/ -v -m "integration"

test-performance:
	@echo "Running performance tests..."
	@$(BIN)/pytest tests/performance/ -v -m "performance"

test-specialized:
	@echo "Running specialized tests..."
	@$(BIN)/pytest tests/specialized/ -v -m "specialized"

test-coverage:
	@echo "Running tests with coverage..."
	@$(BIN)/pytest tests/ --cov=src --cov-report=html --cov-report=xml

.PHONY: lint
lint:
	@echo "Running linters..."
	@$(BIN)/ruff check .
	@$(BIN)/black --check .
	@$(BIN)/isort --check-only .
	@$(BIN)/mypy src/

.PHONY: format
format:
	@echo "Formatting code..."
	@$(BIN)/black .
	@$(BIN)/isort .
	@$(BIN)/ruff --fix .

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
