"""Tests for the LlamaHome GUI interface."""

import json
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

from src.interfaces.gui import GUI, MainWindow


@pytest.fixture
def mock_request_handler():
    """Mock request handler fixture."""
    handler = MagicMock()
    handler.process_request.return_value = "Test response"
    return handler


@pytest.fixture
def gui_interface(mock_request_handler):
    """GUI interface fixture."""
    with patch.object(QApplication, '__init__', return_value=None):
        gui = GUI(request_handler=mock_request_handler)
        yield gui
        gui.stop()


def test_main_window_init(mock_request_handler):
    """Test MainWindow initialization."""
    config = {"model_path": "test/model.bin", "temperature": 0.7}
    window = MainWindow(mock_request_handler, config)
    
    assert window.windowTitle() == "LlamaHome"
    assert window.config == config
    assert window.request_handler == mock_request_handler
    assert isinstance(window.history, list)


def test_submit_prompt(gui_interface):
    """Test prompt submission."""
    window = gui_interface.window
    window.input_text.setPlainText("Test prompt")
    
    window._submit_prompt()
    
    assert window.request_handler.process_request.called
    assert window.request_handler.process_request.call_args[0][0] == "Test prompt"
    assert len(window.history) == 1
    assert window.history[0][0] == "Test prompt"


def test_save_load_history(gui_interface, tmp_path):
    """Test saving and loading conversation history."""
    window = gui_interface.window
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


def test_error_handling(gui_interface):
    """Test error handling in GUI."""
    window = gui_interface.window
    window.request_handler.process_request.side_effect = Exception("Test error")
    
    with patch.object(QMessageBox, 'critical') as mock_error:
        window._submit_prompt()
        mock_error.assert_called_once()
        assert "Test error" in mock_error.call_args[0][1]
