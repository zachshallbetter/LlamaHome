.PHONY: setup run test check clean benchmark needle-test model setup-clean run-clean help code-check type-check

# Detect OS and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Detect Python version
PYTHON_VERSION := $(shell python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Common variables
WORKSPACE_ROOT := $(shell pwd)
PYTHON_CACHE_PREFIX := export PYTHONPYCACHEPREFIX="$(WORKSPACE_ROOT)/.cache/pycache"
SKIP_CHECKS := export LLAMAHOME_SKIP_MEMORY_CHECK=1 && export LLAMAHOME_SKIP_GPU_CHECK=1

# Format code before running checks
format:
	@echo "Formatting code..."
	poetry run isort src tests utils
	poetry run black src tests utils
	poetry run autopep8 --in-place --recursive src tests utils

# Run code quality checks
code-check:
	@echo "Running code quality checks..."
	poetry run python -m utils.code_check

# Run type checks
type-check:
	@echo "Running type checks..."
	poetry run mypy --config-file .config/llamahome.types.ini src tests utils

# Run all checks
check: code-check type-check
	@echo "Running system checks..."
	poetry run python -m utils.system_check

poetry-setup:
	@if [ "$(PYTHON_VERSION)" != "3.11" ]; then \
		echo "ERROR: Python 3.11 is required. Current version: $(PYTHON_VERSION)"; \
		echo "TIP: You can switch Python version with: poetry env use python3.11"; \
		exit 1; \
	fi
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		poetry lock --no-update; \
	else \
		poetry config repositories.pytorch https://download.pytorch.org/whl/cu121; \
		poetry lock --no-update; \
	fi
	@poetry install --quiet

clean-pycache:
	@echo "Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf .cache/pycache

clean-all-cache: clean-pycache
	@echo "Cleaning all caches..."
	rm -rf .cache/*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf code_check_results.txt

clean: clean-all-cache
	@echo "Cleaning up..."

setup-base: check
	@echo "Setting up LlamaHome..."
	$(SKIP_CHECKS) && \
	$(PYTHON_CACHE_PREFIX) && \
	poetry run python setup.py
	@if [ $$? -ne 0 ]; then \
		echo "Error: Setup failed. Check logs for details."; \
		exit 1; \
	fi
	@echo "Setup complete! Run 'make run' to start LlamaHome"

setup: clean-pycache setup-base

setup-clean: clean-all-cache setup-base

run-base:
	@echo "Starting LlamaHome..."
	@$(PYTHON_CACHE_PREFIX) && \
	poetry run python run.py
	@if [ $$? -ne 0 ]; then \
		echo "Error: LlamaHome failed to start. Check logs for details."; \
		exit 1; \
	fi

run: clean-pycache run-base

run-clean: clean-all-cache run-base

test:
	@echo "Running tests..."
	poetry run pytest --runslow --gpu --integration --benchmark --needle

benchmark:
	@echo "Running performance benchmarks..."
	poetry run pytest --benchmark

needle-test:
	@echo "Running needle-in-haystack tests..."
	poetry run pytest --needle

model:
	@echo "Setting up model..."
	@poetry run python -m utils.setup_model \
		--model $(or $(MODEL),llama) \
		--version $(or $(VERSION),latest)

help:
	@echo "Available commands:"
	@echo "  make setup      - Set up LlamaHome (normal)"
	@echo "  make setup -c   - Set up LlamaHome with clean caches"
	@echo "  make run       - Run LlamaHome (normal)"
	@echo "  make run -c    - Run LlamaHome with clean caches"
	@echo "  make clean     - Clean all caches and temporary files"
	@echo "  make test      - Run all tests"
	@echo "  make check     - Run all code checks"
	@echo "  make code-check - Run code quality checks"
	@echo "  make type-check - Run type checks"
	@echo "  make benchmark - Run performance benchmarks"
	@echo "  make model     - Set up a specific model"
	@echo "  make help      - Show this help message"

# Handle -c flag for setup and run
ifeq ($(findstring -c,$(MAKEFLAGS)),-c)
setup: setup-clean
run: run-clean
endif
