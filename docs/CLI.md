# LlamaHome Command-Line Interface (CLI)

This document provides detailed instructions for using the LlamaHome CLI to interact with
the Llama model. The CLI offers a modern shell-like interface with rich features for both
interactive and scripted use.

## Overview

Key Features:

- Shell-like command interface with history and completion
- Persistent command history across sessions
- Dynamic auto-completion and suggestions
- Advanced key bindings and mouse support
- Asynchronous request processing
- Multiple output formats
- Progress indicators
- Configurable timeouts

## Quick Start

### Prerequisites

Required components:

- Python 3.11 or higher
- Poetry (dependency management)
- Llama model

### Installation

1. Install Poetry if you haven't already:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone and install LlamaHome:

   ```bash
   # Clone repository
   git clone https://github.com/llamahome/llamahome.git
   cd llamahome

   # Install dependencies with Poetry
   poetry install

   # Activate virtual environment
   poetry shell
   ```

3. Set up environment:

   ```bash
   # Set model path
   export LLAMA_HOME_MODEL_PATH=/path/to/llama/model  # Unix
   set LLAMA_HOME_MODEL_PATH=C:\path\to\llama\model   # Windows
   ```

## Usage

### Shell Features

#### Command History
- Up/Down arrows to navigate through previous commands
- History persisted in `.config/history.txt`
- Ctrl+R to search through command history

#### Auto-completion
- Tab completion for commands and arguments
- Dynamic completion based on available models
- Multi-column completion menu
- Completion while typing

#### Key Bindings
- Ctrl+C: Cancel current operation
- Ctrl+D: Exit CLI
- Left/Right arrows: Cursor movement
- Home/End: Jump to start/end of line
- Ctrl+K: Cut to end of line
- Ctrl+U: Cut to beginning of line
- Ctrl+W: Delete previous word
- Ctrl+Y: Paste previously cut text

#### Mouse Support
- Click to position cursor
- Click to select completion options
- Scroll through completion menu

#### Auto-suggestions
- Gray text suggestions based on history
- Right arrow to accept suggestion

### Basic Commands

- `help`: Display help information
- `models`: List available models
- `model`: Select a model
- `download`: Download a model
- `remove`: Remove a model
- `chat`: Start an interactive chat session
- `train`: Train a model
- `quit`: Exit CLI
