"""User interface components for LlamaHome.

This package contains the user interface components:
- CLI: Command-line interface for text-based interaction
- GUI: Graphical user interface for visual interaction
- API: REST API interface for programmatic interaction

Each interface provides:
- User input handling
- Response display
- Configuration management
- Session history tracking
"""

from typing import Type, Union

from .cli import CLIInterface
from .gui import GUI

# Type aliases
Interface = Union[CLIInterface, GUI]
InterfaceType = Type[Union[CLIInterface, GUI]]

__all__ = [
    # Classes
    "CLIInterface",
    "GUI",
    "app",
    # Type aliases
    "Interface",
    "InterfaceType",
]
