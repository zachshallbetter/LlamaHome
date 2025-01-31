#!/usr/bin/env python3
"""Installation script with security measures."""

import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def validate_command(command: List[str]) -> None:
    """Validate command for security.

    Args:
        command: Command list to validate

    Raises:
        ValueError: If command is invalid
    """
    if not command:
        raise ValueError("Empty command")

    # Validate executable
    executable = command[0]
    if not Path(executable).is_file():
        # Check PATH
        found = False
        for path in os.environ.get("PATH", "").split(os.pathsep):
            if Path(path) / executable:
                found = True
                break
        if not found:
            raise ValueError(f"Invalid executable: {executable}")

    # Validate arguments
    for arg in command[1:]:
        if not isinstance(arg, str):
            raise ValueError(f"Invalid argument type: {type(arg)}")
        if ";" in arg or "|" in arg or "&" in arg:
            raise ValueError(f"Invalid argument characters in: {arg}")


def run_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run command securely.

    Args:
        command: Command list to run
        cwd: Optional working directory

    Returns:
        Tuple of (return code, stdout, stderr)

    Raises:
        subprocess.CalledProcessError: If command fails
    """
    # Validate command
    validate_command(command)

    # Run command securely
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.stderr}", file=sys.stderr)
        raise


def is_cuda_available() -> bool:
    """Check if CUDA is available and properly configured.

    Returns:
        Whether CUDA is available
    """
    if platform.system() == "Darwin":  # macOS
        return False

    # Check for CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        return False

    # Check for nvcc
    try:
        subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon.

    Returns:
        Whether running on Apple Silicon
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_torch_install_command() -> List[str]:
    """Get the appropriate torch installation command for the platform.

    Returns:
        Installation command list
    """
    base_cmd = [sys.executable, "-m", "pip"]
    if is_apple_silicon():
        return base_cmd + [
            "install",
            "--pre",
            "torch",
            "torchvision",
            "torchaudio",
            "--extra-index-url",
            "https://download.pytorch.org/whl/nightly/cpu",
        ]
    return base_cmd + ["install", "torch"]


def setup_directories() -> None:
    """Create required directories securely."""
    directories = [
        ".cache/models",
        ".cache/training",
        ".cache/system",
        ".data/models",
        ".data/training",
        ".data/metrics",
        ".logs",
        "config",
    ]
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        # Set secure permissions
        path.chmod(0o750)


def setup_config_files() -> None:
    """Set up configuration files securely."""
    # Copy example files if they don't exist
    example_files = [
        (".env.example", ".env"),
        ("config/training_config.toml.example", "config/training_config.toml"),
        ("config/models.json.example", "config/models.json"),
    ]
    for src, dst in example_files:
        src_path = Path(src)
        dst_path = Path(dst)
        if not dst_path.exists() and src_path.exists():
            dst_path.write_text(src_path.read_text())
            # Set secure permissions
            dst_path.chmod(0o640)


def install_dependencies() -> None:
    """Install Python dependencies securely."""
    requirements = Path(__file__).parent.parent / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError("requirements.txt not found")

    print("Installing Python dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    # Install appropriate PyTorch version
    print("Installing PyTorch...")
    run_command(get_torch_install_command())

    # Install development dependencies if in dev mode
    if os.environ.get("LLAMAHOME_ENV") == "development":
        print("Installing development dependencies...")
        run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev,test]"])


def setup_environment() -> None:
    """Set up development environment securely."""
    # Create virtual environment
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])

    # Create directory structure
    print("Creating directory structure...")
    setup_directories()

    # Set up configuration files
    print("Setting up configuration files...")
    setup_config_files()

    # Install dependencies
    install_dependencies()

    # Verify installation
    verify_installation()

    print("Environment setup complete!")


def verify_installation() -> None:
    """Verify installation and print system information."""
    print("\nSystem Information:")
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    if is_cuda_available():
        print("CUDA: Available")
        try:
            import torch
            print(f"PyTorch CUDA: {torch.version.cuda}")
        except ImportError:
            print("PyTorch: Not installed")
    else:
        print("CUDA: Not available")
        if is_apple_silicon():
            print("Apple Silicon detected: Using MPS backend")
        else:
            print("Running in CPU-only mode")


if __name__ == "__main__":
    try:
        setup_environment()
    except Exception as e:
        print(f"Setup failed: {e}", file=sys.stderr)
        sys.exit(1)
