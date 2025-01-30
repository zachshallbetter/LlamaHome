"""System requirements checking utilities."""

import sys
from typing import Dict

import torch

from src.core.utils.log_manager import LogManager, LogTemplates


class SystemCheck:
    """Checks system requirements and capabilities."""

    def __init__(self) -> None:
        """Initialize system checker."""
        self.logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

    def check_requirements(self) -> Dict[str, bool]:
        """Check if system meets requirements.

        Returns:
            Dictionary of requirement check results
        """
        results = {
            "python_version": self._check_python_version(),
            "gpu_available": self._check_gpu_available(),
            "memory_sufficient": self._check_memory_sufficient(),
        }

        self._log_results(results)
        return results

    def _check_python_version(self) -> bool:
        """Check if Python version meets requirements.

        Returns:
            True if version requirements are met
        """
        version = sys.version_info
        return version.major == 3 and version.minor >= 11

    def _check_gpu_available(self) -> bool:
        """Check if GPU acceleration is available.

        Returns:
            True if GPU is available
        """
        return bool(torch.cuda.is_available() or torch.backends.mps.is_available())

    def _check_memory_sufficient(self) -> bool:
        """Check if system has sufficient memory.

        Returns:
            True if memory requirements are met
        """
        import psutil

        total_gb = psutil.virtual_memory().total / (1024**3)
        return bool(total_gb >= 16)  # Require at least 16GB RAM

    def _log_results(self, results: Dict[str, bool]) -> None:
        """Log requirement check results.

        Args:
            results: Dictionary of check results
        """
        for check, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            self.logger.info(f"System check - {check}: {status}")

    def check_system_requirements(self) -> bool:
        """Check if all system requirements are met.

        Returns:
            True if all requirements are met, False otherwise
        """
        results = self.check_requirements()
        return all(results.values())
