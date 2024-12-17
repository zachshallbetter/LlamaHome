"""Tests for data storage functionality."""

from pathlib import Path
from typing import Any, Dict

import pytest

from src.data.storage import DataStorage, create_storage


@pytest.fixture
async def storage(tmp_path: Path) -> DataStorage:
    """Create a temporary storage instance."""
    storage = create_storage(base_path=tmp_path)
    yield storage
    # Cleanup
    await storage.clear_cache()


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Create sample data for testing."""
    return {
        "text": "Sample training text",
        "metadata": {
            "source": "test",
            "timestamp": "2024-01-01T00:00:00"
        }
    }


@pytest.mark.asyncio
async def test_store_and_load_data(storage: DataStorage, sample_data: Dict[str, Any]) -> None:
    """Test storing and loading data."""
    # Store data
    file_path = storage.base_path / "test.json"
    stored_path = await storage.store_data(sample_data, file_path)
    assert stored_path.exists()

    # Load data
    loaded_data = await storage.load_data(stored_path)
    assert loaded_data == sample_data


@pytest.mark.asyncio
async def test_compressed_storage(storage: DataStorage, sample_data: Dict[str, Any]) -> None:
    """Test compressed data storage."""
    file_path = storage.base_path / "test.json"
    stored_path = await storage.store_data(sample_data, file_path, compress=True)
    assert stored_path.suffix == '.gz'
    assert stored_path.exists()


@pytest.mark.asyncio
async def test_cache_functionality(storage: DataStorage, sample_data: Dict[str, Any]) -> None:
    """Test data caching."""
    file_path = storage.base_path / "test.json"
    await storage.store_data(sample_data, file_path)

    # First load should cache
    data1 = await storage.load_data(file_path, use_cache=True)
    assert str(file_path) in storage.cache

    # Second load should use cache
    data2 = await storage.load_data(file_path, use_cache=True)
    assert data1 == data2

    # Clear cache
    await storage.clear_cache()
    assert str(file_path) not in storage.cache


@pytest.mark.asyncio
async def test_archive_data(storage: DataStorage, sample_data: Dict[str, Any]) -> None:
    """Test data archiving."""
    # Store original file
    source_path = storage.base_path / "original.json"
    await storage.store_data(sample_data, source_path)

    # Archive the file
    archive_path = await storage.archive_data(source_path)
    assert archive_path.exists()
    assert archive_path.parent == storage.archive_dir


@pytest.mark.asyncio
async def test_list_data(storage: DataStorage, sample_data: Dict[str, Any]) -> None:
    """Test listing data files."""
    # Create multiple files
    for i in range(3):
        file_path = storage.base_path / f"test_{i}.json"
        await storage.store_data(sample_data, file_path)

    # List all files
    files = await storage.list_data()
    assert len(files) == 3
    assert all(f.suffix == '.json' for f in files)


@pytest.mark.asyncio
async def test_storage_info(storage: DataStorage, sample_data: Dict[str, Any]) -> None:
    """Test storage information retrieval."""
    # Store some data
    file_path = storage.base_path / "test.json"
    await storage.store_data(sample_data, file_path)

    # Get storage info
    info = await storage.get_storage_info()
    assert info["base_path"] == str(storage.base_path)
    assert info["total_size_bytes"] > 0
    assert "cache_entries" in info


@pytest.mark.asyncio
async def test_error_handling(storage: DataStorage) -> None:
    """Test error handling for invalid operations."""
    # Test loading non-existent file
    with pytest.raises(FileNotFoundError):
        await storage.load_data("nonexistent.json")

    # Test storing to invalid path
    with pytest.raises(Exception):
        await storage.store_data({}, "/invalid/path/file.json")


@pytest.mark.asyncio
async def test_directory_structure(storage: DataStorage) -> None:
    """Test directory structure creation."""
    assert storage.training_data.exists()
    assert storage.cache_dir.exists()
    assert storage.archive_dir.exists()
