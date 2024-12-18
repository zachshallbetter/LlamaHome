"""Environment setup and configuration."""

import platform
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

from .utils import LogManager, LogTemplates



logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


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
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False

    try:

        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except ImportError:
        return False


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


def get_device_properties() -> Dict[str, any]:
    """Get properties of the compute device.

    Returns:
        Dictionary of device properties
    """
    device = get_compute_device()
    props = {
        "device_type": device,
        "device_count": 1,
        "memory_allocated": 0,
        "memory_reserved": 0
    }

    if device == "cuda":
        props.update({
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved()
        })
    elif device == "mps":
        # MPS doesn't provide detailed memory info
        props.update({
            "device_name": "Apple Silicon",
            "memory_allocated": -1,  # Not available for MPS
            "memory_reserved": -1    # Not available for MPS
        })

    return props


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

    # Set compute device and properties
    device = get_compute_device()
    env_settings["COMPUTE_DEVICE"] = device
    env_settings["DEVICE_PROPERTIES"] = str(get_device_properties())

    # Configure PyTorch behavior
    if device == "cuda":
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device == "mps":
        # Configure MPS fallback behavior
        env_settings["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable high watermark
        env_settings["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"   # Disable low watermark

    return env_settings


def check_system_requirements() -> Tuple[bool, str]:
    """Check if system meets all requirements.

    Returns:
        Tuple of (requirements met, error message if not met)
    """
    if not check_python_version():
        return False, "Python version must be 3.11 or higher"

    device = get_compute_device()
    if device == "cpu":
        return False, "No GPU acceleration available (CUDA or MPS required)"

    device_props = get_device_properties()
    logger.info(f"Using device: {device} - {device_props.get('device_name', 'Unknown')}")

    return True, ""
