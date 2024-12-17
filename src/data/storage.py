"""Data storage and management module."""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class DataStorage:
    """Manages data storage and retrieval operations."""

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize storage manager.

        Args:
            base_path: Base directory for data storage
            config: Optional configuration dictionary
        """
        self.base_path = Path(base_path or Path.home() / '.llamahome' / 'data')
        self.config = config or {}
        self.cache: Dict[str, Any] = {}

        # Create storage directories
        self.training_data = self.base_path / 'training'
        self.cache_dir = self.base_path / 'cache'
        self.archive_dir = self.base_path / 'archive'

        for directory in [self.training_data, self.cache_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    async def store_data(
        self,
        data: Union[str, Dict, List],
        file_path: Union[str, Path],
        compress: bool = False
    ) -> Path:
        """Store data to file system.

        Args:
            data: Data to store
            file_path: Target file path
            compress: Whether to compress the data

        Returns:
            Path to stored data
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(file_path, 'w') as f:
                if isinstance(data, (dict, list)):
                    await f.write(json.dumps(data, indent=2))
                else:
                    await f.write(str(data))

            if compress:
                compressed_path = file_path.with_suffix('.gz')
                await asyncio.to_thread(
                    shutil.make_archive,
                    str(file_path),
                    'gztar',
                    str(file_path.parent),
                    str(file_path.name)
                )
                return compressed_path

            return file_path

        except Exception as e:
            logger.error(f"Error storing data to {file_path}: {e}")
            raise

    async def load_data(
        self,
        file_path: Union[str, Path],
        use_cache: bool = True
    ) -> Union[str, Dict[Any, Any], List[Any]]:
        """Load data from file system.

        Args:
            file_path: Path to data file
            use_cache: Whether to use cached data

        Returns:
            Loaded data
        """
        file_path = Path(file_path)

        if use_cache and str(file_path) in self.cache:
            data = self.cache[str(file_path)]
            if isinstance(data, (str, dict, list)):
                return data
            return str(data)

        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()

            try:
                data = json.loads(content)
                if not isinstance(data, (dict, list)):
                    return str(data)
                return data
            except json.JSONDecodeError:
                return content

            if use_cache:
                self.cache[str(file_path)] = data

            return data

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    async def archive_data(
        self,
        source_path: Union[str, Path],
        compress: bool = True
    ) -> Path:
        """Archive data file.

        Args:
            source_path: Path to data file
            compress: Whether to compress archive

        Returns:
            Path to archived data
        """
        source_path = Path(source_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = self.archive_dir / \
            f"{source_path.stem}_{timestamp}{source_path.suffix}"

        try:
            await asyncio.to_thread(shutil.copy2, source_path, archive_path)

            if compress:
                compressed_path = archive_path.with_suffix('.gz')
                await asyncio.to_thread(
                    shutil.make_archive,
                    str(archive_path),
                    'gztar',
                    str(archive_path.parent),
                    str(archive_path.name)
                )
                await asyncio.to_thread(archive_path.unlink)
                return compressed_path

            return archive_path

        except Exception as e:
            logger.error(f"Error archiving {source_path}: {e}")
            raise

    async def clear_cache(self) -> None:
        """Clear cached data."""
        self.cache.clear()
        try:
            await asyncio.to_thread(shutil.rmtree, self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    async def list_data(
        self,
        directory: Optional[Union[str, Path]] = None,
        pattern: str = "*"
    ) -> List[Path]:
        """List available data files.

        Args:
            directory: Optional directory to list
            pattern: Glob pattern for filtering

        Returns:
            List of file paths
        """
        directory = Path(directory) if directory else self.base_path

        try:
            files = list(directory.glob(pattern))
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing data in {directory}: {e}")
            raise

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get storage system information.

        Returns:
            Dictionary with storage information
        """
        try:
            total_size = sum(
                f.stat().st_size for f in self.base_path.rglob('*') if f.is_file()
            )

            return {
                "base_path": str(self.base_path),
                "total_size_bytes": total_size,
                "cache_entries": len(self.cache),
                "config": self.config
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            raise


def create_storage(
    base_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None
) -> DataStorage:
    """Factory function to create a DataStorage instance.

    Args:
        base_path: Optional base directory for storage
        config: Optional configuration dictionary

    Returns:
        DataStorage instance
    """
    return DataStorage(base_path, config)
