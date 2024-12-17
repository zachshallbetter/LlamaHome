"""Cache management utilities.

This module provides centralized cache management for LlamaHome, implementing the
caching strategies outlined in docs/Architecture.md.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, Union

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class CacheConfig:
    """Cache configuration following Architecture.md specifications."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """Initialize cache config.
        
        Args:
            base_path: Optional base directory for cache. If not provided,
                      defaults to .cache in current directory.
        """
        # Base cache directory structure
        self.workspace_root = Path.cwd()
        self.BASE_DIR = self.workspace_root / ".cache"
        self.PYCACHE_DIR = self.BASE_DIR / "pycache"
        
        # Cache directory mapping following Architecture.md
        self.CACHE_DIRS = {
            "models": self.BASE_DIR / "models",      # Model weights and parameters
            "training": self.BASE_DIR / "training",  # Training artifacts
            "system": self.BASE_DIR / "system",      # System-level cache
            "pycache": self.PYCACHE_DIR,            # Python bytecode cache
        }

        # Cache size limits from Architecture.md memory management section
        self.CACHE_LIMITS = {
            "models": 1024 * 1024 * 1024,    # 1GB for model files
            "training": 512 * 1024 * 1024,    # 512MB for training data
            "system": 256 * 1024 * 1024,      # 256MB for system cache
            "pycache": 128 * 1024 * 1024,     # 128MB for bytecode
        }

        # Subdirectory structure matching Architecture.md
        self.SUBDIRS = {
            "models": ["llama", "gpt4", "claude"],
            "training": ["datasets", "metrics"],
            "system": ["pytest", "mypy", "temp"],
            "pycache": [],  # Python manages this structure
        }


class CacheManager:
    """Centralized cache management system implementing Architecture.md specs."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager.

        Args:
            base_path: Optional base directory for cache
            config: Optional configuration overrides
        """
        self.config = CacheConfig(base_path)
        self._setup_directories()
        self._configure_pycache()

    def _setup_directories(self) -> None:
        """Create cache directory structure."""
        try:
            # Create main cache directories
            for cache_dir in self.config.CACHE_DIRS.values():
                cache_dir.mkdir(parents=True, exist_ok=True)

            # Create model-specific subdirectories
            for cache_type, subdirs in self.config.SUBDIRS.items():
                cache_dir = self.config.CACHE_DIRS[cache_type]
                for subdir in subdirs:
                    (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

            logger.info("Cache directory structure initialized")
        except Exception as e:
            logger.error(f"Failed to create cache directories: {e}")
            raise

    def _configure_pycache(self) -> None:
        """Configure centralized Python bytecode cache location."""
        try:
            pycache_dir = str(self.config.PYCACHE_DIR.absolute())
            os.environ["PYTHONPYCACHEPREFIX"] = pycache_dir
            logger.info(f"Centralized pycache directory: {pycache_dir}")
        except Exception as e:
            logger.error(f"Failed to configure pycache: {e}")
            raise

    def clean_cache(self, cache_type: str) -> None:
        """Clean specified cache type.

        Args:
            cache_type: Type of cache to clean (models/training/system/pycache)
        """
        if cache_dir := self.config.CACHE_DIRS.get(cache_type):
            try:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Restore subdirectory structure
                for subdir in self.config.SUBDIRS.get(cache_type, []):
                    (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

                logger.info(f"Cleaned {cache_type} cache")
            except Exception as e:
                logger.warning(f"Failed to clean {cache_type} cache: {e}")

    def get_cache_size(self, cache_type: str) -> int:
        """Get total size of specified cache type.

        Args:
            cache_type: Type of cache to measure

        Returns:
            Size in bytes
        """
        if cache_dir := self.config.CACHE_DIRS.get(cache_type):
            try:
                total_size = 0
                for path in cache_dir.rglob("*"):
                    if path.is_file():
                        total_size += path.stat().st_size
                return total_size
            except Exception as e:
                logger.warning(f"Failed to measure {cache_type} cache: {e}")
                return 0
        return 0

    def check_cache_limits(self) -> Dict[str, bool]:
        """Check if cache sizes are within configured limits.

        Returns:
            Dictionary mapping cache types to limit status (True if within limit)
        """
        status = {}
        for cache_type, limit in self.config.CACHE_LIMITS.items():
            current_size = self.get_cache_size(cache_type)
            status[cache_type] = current_size <= limit

            if not status[cache_type]:
                logger.warning(
                    f"{cache_type} cache exceeds limit: {current_size} > {limit} bytes"
                )

        return status

    def cleanup_old_cache(self, max_age_days: int = 7) -> None:
        """Remove cache files older than specified age.

        Args:
            max_age_days: Maximum age of cache files in days
        """
        import time
        current_time = time.time()

        for cache_dir in self.config.CACHE_DIRS.values():
            try:
                for path in cache_dir.rglob("*"):
                    if path.is_file():
                        age = current_time - path.stat().st_mtime
                        if age > max_age_days * 86400:  # Convert days to seconds
                            path.unlink()
                            logger.info(f"Removed old cache file: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean old cache: {e}")

    def clean_pycache(self) -> None:
        """Clean all Python bytecode cache."""
        try:
            # Clean centralized cache
            self.clean_cache("pycache")

            # Clean project-wide pycache
            project_root = self.config.workspace_root
            for pycache_dir in project_root.rglob("__pycache__"):
                try:
                    shutil.rmtree(pycache_dir)
                    logger.info(f"Removed pycache: {pycache_dir}")
                except Exception as e:
                    logger.warning(f"Could not remove pycache {pycache_dir}: {e}")
        except Exception as e:
            logger.warning(f"Failed to clean pycache: {e}")
