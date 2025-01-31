#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[x]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."

    # Check Python version
    if ! command -v python3.11 &> /dev/null; then
        print_error "Python 3.11 is required but not found."
        echo "Please install Python 3.11 first:"
        echo "Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
        echo "macOS: brew install python@3.11"
        exit 1
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is required but not found."
        echo "Please install Git first:"
        echo "Ubuntu/Debian: sudo apt install git"
        echo "macOS: brew install git"
        exit 1
    fi

    # Check memory
    total_mem=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024}')
    if [ -n "$total_mem" ] && [ "$total_mem" -lt 16000000 ]; then
        print_warning "Less than 16GB RAM detected. Performance may be limited."
    fi

    # Check disk space
    free_space=$(df -k . | awk 'NR==2 {print $4}')
    if [ "$free_space" -lt 51200000 ]; then  # 50GB in KB
        print_warning "Less than 50GB free disk space. You may need more space for models."
    fi
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        python3.11 -m venv .venv
    fi

    # Activate virtual environment
    print_status "Activating virtual environment..."
    source .venv/bin/activate

    # Verify activation
    if [ "$VIRTUAL_ENV" = "" ]; then
        print_error "Failed to activate virtual environment"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Run installation script
    python3.11 scripts/install.py

    if [ $? -ne 0 ]; then
        print_error "Installation failed"
        exit 1
    fi
}

# Configure application
configure_application() {
    print_status "Configuring application..."

    # Setup configuration if not exists
    if [ ! -f .env ]; then
        print_status "Setting up initial configuration..."
        cp .env.example .env
        print_warning "Please edit .env file with your preferred settings"
    fi

    # Create necessary directories
    mkdir -p .data/{models,cache,training,metrics} .logs
}

# Main execution
main() {
    print_status "Starting LlamaHome setup..."

    # Run checks and setup
    check_requirements
    setup_environment
    install_dependencies
    configure_application

    print_status "Setup complete!"
    print_status "Starting application..."
    
    # Start application
    python -m src.interfaces.cli
}

# Run main function
main 