"""Data management module for coordinating data operations."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...core.utils import LogManager, LogTemplates
from ...core.utils.cache import CacheManager

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class DataManager:
    """Coordinates data operations across components."""

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize data manager.

        Args:
            base_path: Base directory for data operations
            config: Optional configuration dictionary
        """
        from ..converter import FormatConverter
        from . import StorageManager
        from ..analyzer import TextAnalyzer
        from ..training import TrainingDataManager

        self.base_path = Path(base_path) if base_path else Path.home() / '.llamahome'
        self.config = config or {}

        # Initialize components
        self.storage = StorageManager(self.base_path / 'storage', self.config)
        self.converter = FormatConverter(self.config)
        self.analyzer = TextAnalyzer()
        self.training = TrainingDataManager(self.base_path / 'training')
        self.cache = CacheManager()

    async def process_file(
        self,
        input_path: Union[str, Path],
        analyze: bool = True,
        cache: bool = True
    ) -> Dict[str, Any]:
        """Process input file through conversion, analysis and storage.

        Args:
            input_path: Path to input file
            analyze: Whether to perform text analysis
            cache: Whether to cache results

        Returns:
            Processing results
        """
        input_path = Path(input_path)
        results: Dict[str, Any] = {}

        try:
            # Convert to JSONL if needed
            if input_path.suffix.lower() != '.jsonl':
                jsonl_path = self.base_path / 'temp' / f"{input_path.stem}.jsonl"
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                self.converter.process_file(input_path, jsonl_path)
                input_path = jsonl_path

            # Store data
            stored_path = await self.storage.store_data(
                await self._read_file(input_path),
                self.base_path / 'processed' / input_path.name,
                compress=False
            )
            results['stored_path'] = stored_path

            # Analyze if requested
            if analyze:
                results['analysis'] = self.analyzer.process_document(str(input_path))

            # Cache if requested
            if cache:
                await self._cache_results(input_path, results)

            return results

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            raise

    async def process_directory(
        self,
        input_dir: Union[str, Path],
        recursive: bool = False,
        analyze: bool = True
    ) -> Dict[str, List[str]]:
        """Process all files in directory.

        Args:
            input_dir: Input directory path
            recursive: Whether to process subdirectories
            analyze: Whether to perform analysis

        Returns:
            Processing results by status
        """
        input_dir = Path(input_dir)
        results = {"successful": [], "failed": []}

        try:
            pattern = "**/*" if recursive else "*"
            for file_path in input_dir.glob(pattern):
                if file_path.is_file():
                    try:
                        await self.process_file(file_path, analyze=analyze)
                        results["successful"].append(str(file_path))
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        results["failed"].append(str(file_path))

            return results

        except Exception as e:
            logger.error(f"Error processing directory {input_dir}: {e}")
            raise

    async def _read_file(self, file_path: Path) -> str:
        """Read file content.

        Args:
            file_path: Path to file

        Returns:
            File content
        """
        try:
            async with asyncio.Lock():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    async def _cache_results(self, file_path: Path, results: Dict[str, Any]) -> None:
        """Cache processing results.

        Args:
            file_path: Original file path
            results: Results to cache
        """
        try:
            cache_path = self.base_path / 'cache' / f"{file_path.stem}_results.json"
            await self.storage.store_data(results, cache_path)
        except Exception as e:
            logger.error(f"Error caching results for {file_path}: {e}")
            # Don't raise - caching failure shouldn't stop processing


def create_data_manager(
    base_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None
) -> DataManager:
    """Create a new DataManager instance.

    Args:
        base_path: Optional base directory for data operations
        config: Optional configuration dictionary

    Returns:
        Configured DataManager instance
    """
    return DataManager(base_path, config)
