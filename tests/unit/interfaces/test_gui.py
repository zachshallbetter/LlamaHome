"""Tests for the LlamaHome GUI interface."""

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.interfaces.gui import MainWindow


@pytest.fixture
def mock_request_handler() -> MagicMock:
    """Mock request handler fixture."""
    handler = MagicMock()
    handler.process_request.return_value = "Test response"
    return handler


@pytest.fixture
def gui_interface(mock_request_handler: MagicMock) -> None:
    """GUI interface fixture."""
    with patch.object(QApplication, '__init__', return_value=None):
        window = MainWindow(mock_request_handler)
        yield window
        window.close()


def test_main_window_init(mock_request_handler: MagicMock) -> None:
    """Test MainWindow initialization."""
    window = MainWindow(mock_request_handler)
    
    assert window.windowTitle() == "LlamaHome"
    assert window.request_handler == mock_request_handler
    assert isinstance(window.history, list)


def test_submit_prompt(gui_interface: MainWindow) -> None:
    """Test prompt submission."""
    window = gui_interface
    window.input_text.setPlainText("Test prompt")
    
    window._submit_prompt()
    
    assert window.request_handler.process_request.called
    assert window.request_handler.process_request.call_args[0][0] == "Test prompt"
    assert len(window.history) == 1
    assert window.history[0][0] == "Test prompt"


def test_save_load_history(gui_interface: MainWindow, tmp_path: str) -> None:
    """Test saving and loading conversation history."""
    window = gui_interface
    history_file = tmp_path / "history.json"
    
    # Add some test history
    window.history = [
        ("Test prompt 1", "Test response 1"),
        ("Test prompt 2", "Test response 2")
    ]
    
    with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName',
               return_value=(str(history_file), None)):
        window._save_history()
    
    assert history_file.exists()
    
    # Clear history and reload
    window.history = []
    with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName',
               return_value=(str(history_file), None)):
        window._load_history()
    
    assert len(window.history) == 2
    assert window.history[0][0] == "Test prompt 1"
    assert window.history[1][1] == "Test response 2"


def test_error_handling(gui_interface: MainWindow) -> None:
    """Test error handling in GUI."""
    window = gui_interface
    window.request_handler.process_request.side_effect = Exception("Test error")
    
    with patch.object(QMessageBox, 'critical') as mock_error:
        window._submit_prompt()
        mock_error.assert_called_once()
        assert "Test error" in mock_error.call_args[0][1]
