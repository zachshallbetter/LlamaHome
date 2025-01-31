#!/usr/bin/env python3
"""Installation script for LlamaHome."""

import os
import platform
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], ignore_errors: bool = False) -> None:
    """Run a command with proper error handling."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if not ignore_errors:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e}")
            sys.exit(1)


def is_cuda_available() -> bool:
    """Check if CUDA is available and properly configured."""
    if platform.system() == "Darwin":  # macOS
        return False
    
    # Check for CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        return False
    
    # Check for nvcc
    try:
        subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_requirements(venv_path: Path) -> None:
    """Install package requirements based on platform."""
    pip_cmd = [str(venv_path / "bin" / "pip")]
    
    # Install base requirements first
    print("Installing base requirements...")
    run_command(pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    run_command(pip_cmd + ["install", "numpy"])  # Install numpy before torch
    run_command(pip_cmd + ["install", "torch"])  # Install torch
    
    # Determine which extras to install
    extras = ["dev", "test"]
    if is_cuda_available():
        print("CUDA detected, installing CUDA dependencies...")
        extras.append("cuda")
    else:
        print("CUDA not detected, skipping CUDA dependencies...")
    
    # Install the package with appropriate extras
    print("Installing LlamaHome with extras...")
    extras_str = f".[{','.join(extras)}]"
    run_command(pip_cmd + ["install", "-e", extras_str], ignore_errors=True)


def main() -> None:
    """Run installation process."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Create and activate virtual environment
    venv_path = project_root / ".venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])

    # Install requirements
    install_requirements(venv_path)

    print("Installation complete!")
    
    if not is_cuda_available():
        print("\nNote: CUDA support is not available. To enable GPU acceleration:")
        print("1. Install NVIDIA CUDA Toolkit 11.7+")
        print("2. Set CUDA_HOME environment variable")
        print("3. Run the installation script again")


if __name__ == "__main__":
    main() 