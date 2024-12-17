"""GUI interface for LlamaHome."""

import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.log_manager import LogManager, LogTemplates

from ..core.request_handler import RequestHandler

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class MainWindow(QMainWindow):
    """Main window for LlamaHome GUI."""

    def __init__(self, request_handler: RequestHandler, config: Optional[Dict[str, Any]] = None):
        """Initialize the main window.

        Args:
            request_handler: Request handler instance
            config: Optional configuration dictionary
        """
        super().__init__()
        self.request_handler = request_handler
        self.config: Dict[str, Any] = config or {}
        self.history: List[Tuple[str, str]] = []

        self.setWindowTitle("LlamaHome")
        self.setGeometry(100, 100, 800, 600)

        self._create_ui()
        self._create_menu()
        self._create_status_bar()
        self._bind_shortcuts()

    def _create_ui(self) -> None:
        """Create the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Input area
        input_layout = QVBoxLayout()
        input_label = QLabel("Input")
        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(100)

        button_layout = QHBoxLayout()
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self._submit_prompt)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_input)

        button_layout.addWidget(submit_button)
        button_layout.addWidget(clear_button)
        button_layout.addStretch()

        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_text)
        input_layout.addLayout(button_layout)

        # Output area
        output_layout = QVBoxLayout()
        output_label = QLabel("Output")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_text)

        layout.addLayout(input_layout)
        layout.addLayout(output_layout)

    def _create_menu(self) -> None:
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        save_action = QAction("Save History", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_history)
        file_menu.addAction(save_action)

        load_action = QAction("Load History", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_history)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        clear_input_action = QAction("Clear Input", self)
        clear_input_action.setShortcut("Ctrl+K")
        clear_input_action.triggered.connect(self._clear_input)
        edit_menu.addAction(clear_input_action)

        clear_output_action = QAction("Clear Output", self)
        clear_output_action.setShortcut("Ctrl+L")
        clear_output_action.triggered.connect(self._clear_output)
        edit_menu.addAction(clear_output_action)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")

        model_settings_action = QAction("Model Settings", self)
        model_settings_action.triggered.connect(self._show_settings)
        settings_menu.addAction(model_settings_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self) -> None:
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts."""
        submit_shortcut = QAction(self)
        submit_shortcut.setShortcut("Ctrl+Return")
        submit_shortcut.triggered.connect(self._submit_prompt)
        self.addAction(submit_shortcut)

    def _submit_prompt(self) -> None:
        """Submit prompt to model."""
        prompt = self.input_text.toPlainText().strip()
        if not prompt:
            self._show_error("Please enter a prompt")
            return

        try:
            self.status_bar.showMessage("Processing...")
            QApplication.processEvents()

            response: Any = self.request_handler.process_request(prompt)
            self.history.append((prompt, response))

            self._append_output(f"Prompt: {prompt}\n")
            self._append_output(f"Response: {response}\n\n")
            self._clear_input()

            self.status_bar.showMessage("Ready")
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            self._show_error(f"Error: {e}")
            self.status_bar.showMessage("Error")

    def _append_output(self, text: str) -> None:
        """Append text to output area."""
        self.output_text.append(text)
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.output_text.setTextCursor(cursor)

    def _clear_input(self) -> None:
        """Clear input area."""
        self.input_text.clear()

    def _clear_output(self) -> None:
        """Clear output area."""
        self.output_text.clear()

    def _save_history(self) -> None:
        """Save conversation history to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save History",
            "",
            "JSON files (*.json);;All files (*.*)",
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(self.history, f, indent=2)
                self.status_bar.showMessage(f"History saved to {file_path}")
            except Exception as e:
                logger.error(f"Error saving history: {e}")
                self._show_error(f"Could not save history: {e}")

    def _load_history(self) -> None:
        """Load conversation history from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load History",
            "",
            "JSON files (*.json);;All files (*.*)",
        )

        if file_path:
            try:
                with open(file_path) as f:
                    self.history = json.load(f)

                self._clear_output()
                for prompt, response in self.history:
                    self._append_output(f"Prompt: {prompt}\n")
                    self._append_output(f"Response: {response}\n\n")

                self.status_bar.showMessage(f"History loaded from {file_path}")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                self._show_error(f"Could not load history: {e}")

    def _show_settings(self) -> None:
        """Show settings dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Model settings
        form_layout = QFormLayout()

        model_path = QLineEdit()
        model_path.setText(self.config.get("model_path", ""))
        form_layout.addRow("Model Path:", model_path)

        temperature = QLineEdit()
        temperature.setText(str(self.config.get("temperature", 0.7)))
        form_layout.addRow("Temperature:", temperature)

        layout.addLayout(form_layout)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.config["model_path"] = model_path.text()
            self.config["temperature"] = float(temperature.text())
            self.status_bar.showMessage("Settings updated")

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About LlamaHome",
            "LlamaHome GUI\n\n"
            "A modern interface for Llama 3.3\n"
            "Version 1.0.0\n\n"
            "© 2024 LlamaHome Team",
        )

    def _show_error(self, message: str) -> None:
        """Show error dialog."""
        QMessageBox.critical(self, "Error", message)


class GUI:
    """GUI wrapper class for backward compatibility."""

    def __init__(self, request_handler: RequestHandler, config: Optional[Dict[str, Any]] = None):
        """Initialize the GUI.

        Args:
            request_handler: Request handler instance
            config: Optional configuration dictionary
        """
        self.app = QApplication(sys.argv)
        self.window = MainWindow(request_handler, config)

    def start(self) -> None:
        """Start the GUI application."""
        try:
            self.window.show()
            sys.exit(self.app.exec())
        except Exception as e:
            logger.error(f"GUI error: {e}")
            raise

    def stop(self) -> None:
        """Stop the GUI application."""
        self.app.quit()


def create_gui(request_handler: RequestHandler, config: Optional[Dict[str, Any]] = None) -> GUI:
    """Create a GUI instance.

    Args:
        request_handler: Request handler instance
        config: Optional configuration dictionary

    Returns:
        GUI instance
    """
    return GUI(request_handler, config)
