"""Environment setup utilities."""

import os
from pathlib import Path
from typing import Dict

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class EnvironmentSetup:
    """Environment setup and configuration manager."""
    
    def __init__(self):
        """Initialize environment setup."""
        self.workspace_root = Path.cwd()
        self.data_dir = self.workspace_root / "data"
        self.models_dir = self.data_dir / "models"
        self.cache_dir = self.workspace_root / ".cache"
        self.logs_dir = self.workspace_root / ".logs"
        self.config_dir = self.workspace_root / ".config"

    def setup_environment(self) -> None:
        """Set up the environment directories and variables."""
        # Create necessary directories
        self._create_directories()
        
        # Set environment variables
        self._set_environment_variables()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.cache_dir,
            self.logs_dir,
            self.config_dir
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")

    def _set_environment_variables(self) -> None:
        """Set required environment variables."""
        env_vars = {
            "LLAMAHOME_ROOT": str(self.workspace_root),
            "LLAMAHOME_DATA": str(self.data_dir),
            "LLAMAHOME_MODELS": str(self.models_dir),
            "LLAMAHOME_CACHE": str(self.cache_dir),
            "LLAMAHOME_LOGS": str(self.logs_dir),
            "LLAMAHOME_CONFIG": str(self.config_dir)
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.debug(f"Set environment variable {key}={value}")

    def get_env_info(self) -> Dict[str, str]:
        """Get environment information.
        
        Returns:
            Dictionary of environment information
        """
        return {
            "Workspace Root": str(self.workspace_root),
            "Data Directory": str(self.data_dir),
            "Models Directory": str(self.models_dir),
            "Cache Directory": str(self.cache_dir),
            "Logs Directory": str(self.logs_dir),
            "Config Directory": str(self.config_dir)
        }
