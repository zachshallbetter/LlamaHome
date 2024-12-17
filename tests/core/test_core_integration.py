"""Integration tests for core functionality."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.core.request_handler import RequestHandler
from src.data.storage import DataStorage
from src.data.training import TrainingDataManager, create_training_manager


@pytest.fixture
async def request_handler() -> RequestHandler:
    """Create a request handler instance for testing."""
    return RequestHandler(model_integration=True)


@pytest.fixture
async def storage(tmp_path: Path) -> DataStorage:
    """Create a temporary storage instance."""
    storage = DataStorage(base_path=tmp_path)
    return storage


@pytest.fixture
async def training_manager(storage: DataStorage) -> TrainingDataManager:
    """Create a training manager instance for testing."""
    manager: TrainingDataManager = await create_training_manager(data_dir=storage.base_path)
    await manager.initialize()
    return manager


@pytest.mark.asyncio
async def test_core_integration(
    request_handler: RequestHandler, storage: DataStorage, training_manager: TrainingDataManager
) -> None:
    """Test core integration."""
    # Test data storage
    test_data: Dict[str, Any] = {"text": "Test content", "metadata": {"source": "integration_test"}}
    file_path = storage.base_path / "test.json"
    await storage.store_data(test_data, file_path)

    # Test data loading
    loaded_data: Dict[str, Any] = await storage.load_data(file_path)
    assert loaded_data == test_data

    # Test request handling
    response: Any = await request_handler.process_request("Test request")
    assert response is not None

    # Test training manager
    training_files: List[Path] = await training_manager.list_training_files()
    assert isinstance(training_files, list)


@pytest.mark.asyncio
async def test_core_integration_with_gui(
    request_handler: RequestHandler, storage: DataStorage, training_manager: TrainingDataManager
) -> None:
    """Test core integration with GUI."""
    # Test request handling with GUI context
    gui_request: Dict[str, Any] = {
        "type": "user_input",
        "content": "Test GUI request",
        "metadata": {"source": "gui"},
    }
    response: Any = await request_handler.process_request(json.dumps(gui_request))
    assert response is not None

    # Test data persistence
    test_data: Dict[str, Any] = {"text": "GUI test content", "metadata": {"source": "gui_test"}}
    file_path = storage.base_path / "gui_test.json"
    await storage.store_data(test_data, file_path)

    # Verify data integrity
    loaded_data: Dict[str, Any] = await storage.load_data(file_path)
    assert loaded_data == test_data

    # Test training integration
    training_status: Dict[str, Any] = await training_manager.get_training_status()
    assert isinstance(training_status, dict)
