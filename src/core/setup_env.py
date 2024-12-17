"""Environment setup and configuration."""

import os
import platform
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .utils import LogManager, LogTemplates


def check_python_version() -> bool:
    """Check if Python version meets requirements.
    
    Returns:
        True if version requirements are met
    """
    version = sys.version_info
    return version.major == 3 and version.minor >= 11


def check_cuda_available() -> bool:
    """Check if CUDA is available.
    
    Returns:
        True if CUDA is available
    """
    return torch.cuda.is_available()


def check_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available.
    
    Returns:
        True if MPS is available
    """
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def get_compute_device() -> str:
    """Get the best available compute device.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if check_cuda_available():
        return "cuda"
    if check_mps_available():
        return "mps"
    return "cpu"


def setup_environment() -> Dict[str, str]:
    """Set up environment variables and paths.
    
    Returns:
        Dictionary of environment settings
    """
    env_settings = {}
    
    # Set up paths
    workspace_root = Path.cwd()
    env_settings["WORKSPACE_ROOT"] = str(workspace_root)
    env_settings["DATA_DIR"] = str(workspace_root / "data")
    env_settings["MODELS_DIR"] = str(workspace_root / "models")
    env_settings["CACHE_DIR"] = str(workspace_root / ".cache")
    
    # Create directories
    for path in [env_settings["DATA_DIR"], env_settings["MODELS_DIR"], env_settings["CACHE_DIR"]]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Set compute device
    env_settings["COMPUTE_DEVICE"] = get_compute_device()
    
    return env_settings


def check_system_requirements() -> Tuple[bool, str]:
    """Check if system meets all requirements.
    
    Returns:
        Tuple of (requirements met, error message if not met)
    """
    if not check_python_version():
        return False, "Python version must be 3.11 or higher"
        
    if not (check_cuda_available() or check_mps_available()):
        return False, "No GPU acceleration available (CUDA or MPS required)"
        
    return True, ""
