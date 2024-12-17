"""System check utilities."""

import platform
import psutil
import shutil
import subprocess
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class SystemCheck:
    """System configuration and requirements checker."""
    
    def __init__(self):
        """Initialize system checker."""
        self.python_version = platform.python_version()
        self.os_name = platform.system()
        self.os_version = platform.release()
        self.architecture = platform.machine()
        self.torch_version = torch.__version__

    def check_python_version(self) -> bool:
        """Check Python version meets requirements."""
        major, minor, _ = map(int, self.python_version.split("."))
        is_valid = (major == 3 and minor >= 9)
        if is_valid:
            logger.info(f"Python version check: Python {major}.{minor} detected")
        return is_valid

    def check_memory(self) -> Tuple[bool, float]:
        """Check system memory."""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        is_sufficient = available_gb >= 8
        if is_sufficient:
            logger.info(f"Memory check: {available_gb:.1f}GB RAM available")
        return is_sufficient, available_gb

    def check_disk_space(self, path: str = ".") -> Tuple[bool, float]:
        """Check available disk space."""
        disk = shutil.disk_usage(path)
        available_gb = disk.free / (1024 ** 3)
        is_sufficient = available_gb >= 10
        if is_sufficient:
            logger.info(f"Disk space check: {available_gb:.1f}GB disk space available")
        return is_sufficient, available_gb

    def check_gpu_support(self) -> Tuple[bool, str]:
        """Check GPU/accelerator support."""
        if torch.cuda.is_available():
            device = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
            return True, device
        elif (self.os_name == "Darwin" and 
              self.architecture == "arm64" and 
              hasattr(torch.backends, "mps") and 
              torch.backends.mps.is_available()):
            return True, "Apple Metal (MPS)"
        else:
            return True, f"CPU ({platform.processor() or self.architecture})"

    def check_dependencies(self) -> List[str]:
        """Check required system dependencies."""
        dependencies = ["git", "make", "python3", "pip3"]
        missing = []
        
        for dep in dependencies:
            try:
                subprocess.run(
                    ["which", dep],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Dependency check - {dep}: Found")
            except subprocess.CalledProcessError:
                missing.append(dep)
                
        return missing

    def check_permissions(self) -> Dict[str, str]:
        """Check directory permissions."""
        paths = {
            "current_dir": ".",
            "home_dir": str(Path.home()),
            "temp_dir": "/tmp"
        }
        
        results = {}
        for name, path in paths.items():
            perms = []
            try:
                test_path = Path(path)
                if test_path.exists():
                    if os.access(path, os.R_OK): perms.append("R")
                    if os.access(path, os.W_OK): perms.append("W")
                    if os.access(path, os.X_OK): perms.append("X")
                    status = "+".join(perms) if perms else "No access"
                    logger.info(f"Permission check - {name}: {status}")
                    results[name] = status
                else:
                    results[name] = "Not found"
            except Exception:
                results[name] = "Error"
                
        return results

    def get_system_info(self) -> Dict[str, str]:
        """Get complete system information.

        Returns:
            Dictionary of system information
        """
        # Run all checks
        python_ok = self.check_python_version()
        mem_ok, mem_gb = self.check_memory()
        disk_ok, disk_gb = self.check_disk_space()
        gpu_ok, gpu_info = self.check_gpu_support()
        missing_deps = self.check_dependencies()
        perms = self.check_permissions()
        
        # Build system info
        system_info = {
            "Operating System": f"{self.os_name} {self.os_version}",
            "Architecture": self.architecture,
            "Python Version": self.python_version,
            "PyTorch Version": self.torch_version,
            "Compute Backend": gpu_info
        }
        
        return system_info
