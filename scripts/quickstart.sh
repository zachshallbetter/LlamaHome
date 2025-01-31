#!/bin/bash
set -e

# Check Python version
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 is required but not found."
    echo "Please install Python 3.11 first:"
    echo "Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "macOS: brew install python@3.11"
    exit 1
fi

# Check Git
if ! command -v git &> /dev/null; then
    echo "Git is required but not found."
    echo "Please install Git first:"
    echo "Ubuntu/Debian: sudo apt install git"
    echo "macOS: brew install git"
    exit 1
fi

# Run installation
echo "Running installation..."
python3.11 scripts/install.py

# Setup configuration if not exists
if [ ! -f .env ]; then
    echo "Setting up initial configuration..."
    cp .env.example .env
    echo "Please edit .env file with your preferred settings"
fi

# Activate virtual environment and start application
echo "Starting application..."
source .venv/bin/activate
python -m src.interfaces.cli 