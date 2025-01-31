#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_error() {
    echo -e "${RED}[x]${NC} $1"
}

# Check Python version
if ! command -v python3.11 &> /dev/null; then
    print_error "Python 3.11 is required but not found."
    echo "Please install Python 3.11 first:"
    echo "Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "macOS: brew install python@3.11"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
python3.11 -m venv .venv

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
python -m pip install -r requirements.txt

# Install development dependencies if in dev mode
if [ "$1" = "--dev" ]; then
    print_status "Installing development dependencies..."
    python -m pip install -e ".[dev]"
fi

# Install test dependencies if in test mode
if [ "$1" = "--test" ]; then
    print_status "Installing test dependencies..."
    python -m pip install -e ".[test]"
fi

print_status "Virtual environment setup complete!" 