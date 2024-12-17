"""Cache management utilities.

This module provides centralized cache management for LlamaHome, implementing
a comprehensive caching system for all application components.

Key Features:
- Centralized cache management
- Configurable cache limits
- Automatic cleanup
- Cache monitoring
- Python bytecode optimization

The cache management system follows a hierarchical structure:
1. Models Cache: Model weights and parameters
2. Training Cache: Training artifacts and datasets
3. System Cache: System-level temporary data
4. Python Cache: Centralized bytecode cache

Performance Considerations:
- Automatic size monitoring
- Age-based cleanup
- Directory structure optimization
- Memory limit enforcement

System Requirements:
- Sufficient disk space for all cache types
- Write permissions for cache directories
- For optimal performance:
    - SSD storage recommended
    - Regular cleanup scheduling
    - Monitoring integration

See Also:
    - src/core/cache.py: Model-specific caching
    - src/data/cache.py: Training data cache
    - docs/Architecture.md: System architecture
    - docs/Performance.md: Performance guidelines

Example:
    >>> # Initialize cache manager
    >>> from utils.cache_manager import CacheManager
    >>> cache_mgr = CacheManager()
    >>> 
    >>> # Clean specific cache type
    >>> cache_mgr.clean_cache("models")
    >>> 
    >>> # Check cache limits
    >>> status = cache_mgr.check_cache_limits()
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, Union

from ...core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class CacheConfig:
    """Cache configuration and directory structure manager.
    
    This class defines the structure and limits for all cache types
    in the system. It implements a hierarchical cache organization
    following the system architecture specifications.
    
    Features:
        - Configurable base paths
        - Predefined directory structure
        - Size limits per cache type
        - Subdirectory organization
        
    Directory Structure:
        .cache/
        ├── models/          # Model artifacts
        │   ├── llama/
        │   ├── gpt4/
        │   └── claude/
        ├── training/        # Training data
        │   ├── datasets/
        │   └── metrics/
        ├── system/          # System cache
        │   ├── pytest/
        │   ├── mypy/
        │   └── temp/
        └── pycache/         # Python bytecode
    
    Memory Management:
        Each cache type has a configured size limit:
        - models: 1GB for model files
        - training: 512MB for training data
        - system: 256MB for system cache
        - pycache: 128MB for bytecode
        
    Example:
        >>> config = CacheConfig()
        >>> print(config.CACHE_DIRS["models"])
        PosixPath('.cache/models')
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """Initialize cache configuration.
        
        Args:
            base_path: Optional base directory for cache.
                If not provided, uses .cache in workspace root.
                The directory will be created if it doesn't exist.
                
        Directory Structure:
            Creates a hierarchical cache structure with:
            1. Base cache directory (.cache)
            2. Type-specific directories (models, training, etc.)
            3. Purpose-specific subdirectories
            
        Cache Types:
            - models: Model weights and configurations
            - training: Training data and metrics
            - system: System-level temporary data
            - pycache: Centralized Python bytecode
        """
        # Base cache directory structure
        self.workspace_root = Path.cwd()
        self.BASE_DIR = self.workspace_root / ".cache"
        self.PYCACHE_DIR = self.BASE_DIR / "pycache"
        
        # Cache directory mapping with purpose
        self.CACHE_DIRS = {
            "models": self.BASE_DIR / "models",      # Model weights and parameters
            "training": self.BASE_DIR / "training",  # Training artifacts
            "system": self.BASE_DIR / "system",      # System-level cache
            "pycache": self.PYCACHE_DIR,            # Python bytecode cache
        }

        # Cache size limits with rationale
        self.CACHE_LIMITS = {
            "models": 1024 * 1024 * 1024,    # 1GB: Typical model size range
            "training": 512 * 1024 * 1024,    # 512MB: Dataset cache
            "system": 256 * 1024 * 1024,      # 256MB: Temporary files
            "pycache": 128 * 1024 * 1024,     # 128MB: Bytecode cache
        }

        # Subdirectory structure with purpose
        self.SUBDIRS = {
            "models": ["llama", "gpt4", "claude"],  # Model-specific caches
            "training": ["datasets", "metrics"],     # Training artifacts
            "system": ["pytest", "mypy", "temp"],   # Tool-specific caches
            "pycache": [],  # Managed by Python runtime
        }


class CacheManager:
    """Centralized cache management system.
    
    This class provides comprehensive cache management for all
    system components, implementing:
    - Directory structure management
    - Cache size monitoring
    - Automatic cleanup
    - Python bytecode optimization
    
    Features:
        - Centralized configuration
        - Automatic directory creation
        - Size limit enforcement
        - Age-based cleanup
        - Cache statistics
        
    Thread Safety:
        Basic operations are thread-safe. For concurrent
        cache modifications, use external synchronization.
        
    Example:
        >>> manager = CacheManager()
        >>> manager.clean_cache("models")  # Clean model cache
        >>> manager.check_cache_limits()   # Monitor sizes
        >>> manager.cleanup_old_cache(7)   # Remove old files
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager.

        Args:
            base_path: Optional base directory for cache.
                If not provided, uses default location.
            config: Optional configuration overrides.
                Can customize limits and directories.
                
        Initialization Process:
            1. Creates cache configuration
            2. Sets up directory structure
            3. Configures Python bytecode location
            
        Raises:
            OSError: If cache directories cannot be created
            PermissionError: If lacking write permissions
        """
        self.config = CacheConfig(base_path)
        self._setup_directories()
        self._configure_pycache()

    def _setup_directories(self) -> None:
        """Create cache directory structure.
        
        Creates the complete cache directory hierarchy:
        1. Main cache directories
        2. Type-specific subdirectories
        3. Purpose-specific folders
        
        Directory Creation:
            - Creates parent directories if needed
            - Handles existing directories safely
            - Maintains proper permissions
            
        Raises:
            OSError: If directory creation fails
            PermissionError: If lacking permissions
        """
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
        """Configure centralized Python bytecode cache.
        
        Sets up a centralized location for Python bytecode cache
        to improve performance and maintainability.
        
        Configuration:
            - Sets PYTHONPYCACHEPREFIX environment variable
            - Creates dedicated pycache directory
            - Ensures proper permissions
            
        Benefits:
            - Improved import performance
            - Easier cache cleanup
            - Better organization
            
        Raises:
            OSError: If pycache setup fails
            EnvironmentError: If env var cannot be set
        """
        try:
            pycache_dir = str(self.config.PYCACHE_DIR.absolute())
            os.environ["PYTHONPYCACHEPREFIX"] = pycache_dir
            logger.info(f"Centralized pycache directory: {pycache_dir}")
        except Exception as e:
            logger.error(f"Failed to configure pycache: {e}")
            raise

    def clean_cache(self, cache_type: str) -> None:
        """Clean specified cache type.
        
        Removes all files from the specified cache directory
        and recreates the directory structure.
        
        Args:
            cache_type: Type of cache to clean:
                - "models": Model artifacts
                - "training": Training data
                - "system": System cache
                - "pycache": Python bytecode
                
        Process:
            1. Removes entire cache directory
            2. Recreates directory structure
            3. Restores subdirectories
            
        Thread Safety:
            This operation is not atomic. Use external
            synchronization for concurrent access.
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
        
        Calculates the total size of all files in the
        specified cache directory and its subdirectories.
        
        Args:
            cache_type: Type of cache to measure:
                - "models": Model artifacts
                - "training": Training data
                - "system": System cache
                - "pycache": Python bytecode
                
        Returns:
            Total size in bytes
            
        Performance:
            - O(n) where n is number of files
            - May be slow for large caches
            - Consider using sampling for monitoring
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
        
        Monitors the size of each cache type and compares
        against configured limits.
        
        Returns:
            Dictionary mapping cache types to limit status:
                - key: Cache type string
                - value: True if within limit, False if exceeded
                
        Example:
            >>> status = manager.check_cache_limits()
            >>> if not status["models"]:
            ...     manager.clean_cache("models")
                
        Performance:
            - O(n) where n is total number of files
            - Consider periodic scheduled checks
            - Use sampling for large caches
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
        
        Performs age-based cleanup of cache files across
        all cache types.
        
        Args:
            max_age_days: Maximum age of cache files in days.
                Files older than this are removed.
                
        Process:
            1. Calculates file age from mtime
            2. Removes files exceeding age limit
            3. Preserves directory structure
            
        Thread Safety:
            This operation is not atomic. Use external
            synchronization for concurrent access.
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
        """Clean all Python bytecode cache.
        
        Removes all Python bytecode cache files from:
        1. Centralized cache directory
        2. Project-wide __pycache__ directories
        
        Process:
            1. Cleans centralized cache
            2. Recursively removes __pycache__
            3. Recreates necessary directories
            
        Benefits:
            - Forces Python to recompile modules
            - Removes outdated bytecode
            - Frees disk space
            
        Thread Safety:
            This operation is not atomic. Use external
            synchronization for concurrent access.
        """
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
