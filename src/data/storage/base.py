"""Base storage functionality."""

import json
from pathlib import Path
from typing import Optional, Union, Dict, List

try:
    import aiofiles


    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from src.core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class BaseStorage:
    """Base class for storage implementations."""


    def __init__(self, base_path: Union[str, Path]):
        """Initialize storage.

        Args:
            base_path: Base storage path
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        if not AIOFILES_AVAILABLE:
            logger.warning("aiofiles not available, falling back to synchronous I/O")

    async def read_file(self, relative_path: Union[str, Path]) -> Optional[str]:
        """Read file contents asynchronously.

        Args:
            relative_path: Path relative to base path

        Returns:
            File contents or None if read fails
        """
        try:
            full_path = self.base_path / relative_path

            if AIOFILES_AVAILABLE:
                async with aiofiles.open(full_path, 'r') as f:
                    return await f.read()
            else:
                with open(full_path, 'r') as f:
                    return f.read()

        except Exception as e:
            logger.error(f"Failed to read file {relative_path}: {e}")
            return None

    async def write_file(self, relative_path: Union[str, Path], content: str) -> bool:
        """Write file contents asynchronously.

        Args:
            relative_path: Path relative to base path
            content: Content to write

        Returns:
            True if write succeeds
        """
        try:
            full_path = self.base_path / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if AIOFILES_AVAILABLE:
                async with aiofiles.open(full_path, 'w') as f:
                    await f.write(content)
            else:
                with open(full_path, 'w') as f:
                    f.write(content)

            return True

        except Exception as e:
            logger.error(f"Failed to write file {relative_path}: {e}")
            return False

    async def read_json(self, relative_path: Union[str, Path]) -> Optional[Dict]:
        """Read JSON file asynchronously.

        Args:
            relative_path: Path relative to base path

        Returns:
            Parsed JSON or None if read fails
        """
        content = await self.read_file(relative_path)
        if content is not None:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {relative_path}: {e}")
        return None

    async def write_json(self, relative_path: Union[str, Path], data: Dict) -> bool:
        """Write JSON file asynchronously.

        Args:
            relative_path: Path relative to base path
            data: Data to write

        Returns:
            True if write succeeds
        """
        try:
            content = json.dumps(data, indent=2)
            return await self.write_file(relative_path, content)
        except Exception as e:
            logger.error(f"Failed to write JSON to {relative_path}: {e}")
            return False

    async def list_files(self, relative_path: Union[str, Path] = "") -> List[Path]:
        """List files in directory asynchronously.

        Args:
            relative_path: Optional path relative to base path

        Returns:
            List of file paths
        """
        try:
            full_path = self.base_path / relative_path
            if not full_path.exists():
                return []

            return [
                path.relative_to(self.base_path)
                for path in full_path.rglob("*")
                if path.is_file()
            ]

        except Exception as e:
            logger.error(f"Failed to list files in {relative_path}: {e}")
            return []

    async def delete_file(self, relative_path: Union[str, Path]) -> bool:
        """Delete file asynchronously.

        Args:
            relative_path: Path relative to base path

        Returns:
            True if deletion succeeds
        """
        try:
            full_path = self.base_path / relative_path
            if full_path.exists():
                full_path.unlink()
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {relative_path}: {e}")
            return False
