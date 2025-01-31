# LlamaHome Dependencies

## Table of Contents

- [Core Dependencies](#core-dependencies)
- [AI and Machine Learning](#ai-and-machine-learning)
- [Document Processing](#document-processing)
- [File Handling and I/O](#file-handling-and-io)
- [CLI and Configuration](#cli-and-configuration)
- [Type Stubs](#type-stubs)
- [Development Dependencies](#development-dependencies)
- [GUI Dependencies](#gui-dependencies)

## Overview

This document provides a comprehensive list of dependencies for LlamaHome, including core dependencies, AI and machine learning libraries, document processing tools, file handling and I/O libraries, CLI and configuration utilities, type stubs, development dependencies, and GUI dependencies.

## Core Dependencies

### AI and Machine Learning

- `torch`: PyTorch for deep learning operations (used in H2O implementation)
  - Version: 2.2.1
  - Requires: Python >=3.11,<3.13
  - GPU Support: CUDA 12.1 pre-built
- `transformers`: Hugging Face transformers for Llama 3.3 integration (4.38.2)
- `tokenizers`: Fast tokenization for text processing (0.15.2)
- `h2o`: Heavy-Hitter Oracle implementation (3.48.0)
- `rouge`: ROUGE score evaluation for model outputs (1.0.1)
- `psutil`: System resource monitoring for benchmarks (5.9.8)

### Document Processing

- `nltk`: Natural language processing utilities (3.8.1)
- `pandas`: Data manipulation and analysis (2.2.1)
- `numpy`: Numerical computing (1.26.4)
- `PyMuPDF`: PDF document processing (1.24.0)
- `python-docx`: Microsoft Word document handling (1.1.0)
- `openpyxl`: Excel file processing (3.1.2)
- `python-pptx`: PowerPoint file processing (0.6.23)
- `ebooklib`: E-book format support (0.18.1)
- `striprtf`: RTF document processing (0.0.28)
- `olefile`: Legacy Microsoft Office format support (0.47)
- `odfpy`: OpenDocument format support (1.4.1)
- `ezodf`: Additional OpenDocument support (0.3.2)
- `lxml`: XML and HTML processing (5.1.0)
- `beautifulsoup4`: Web content parsing (4.12.3)
- `console`: Console utilities (0.9909)

### File Handling and I/O

- `aiofiles`: Asynchronous file operations (23.2.1)
- `xopen`: Efficient compressed file handling (2.0.2)
- `markdown`: Markdown processing (3.5.2)

### CLI and Configuration

- `click`: Command line interface creation (8.1.7)
- `python-dotenv`: Environment variable management (1.0.1)
- `pyyaml`: YAML configuration handling (6.0.1)
- `rich`: Rich text and formatting for CLI (13.7.1)
- `requests`: HTTP client for downloads (2.31.0)
- `tqdm`: Progress bars (4.66.2)
- `structlog`: Structured logging (24.4.0)

### Type Stubs

- `types-requests`: Type stubs for requests (2.31.0.20240311)
- `types-tqdm`: Type stubs for tqdm (4.66.0.20240311)
- `types-PyYAML`: Type stubs for PyYAML (6.0.12.12)
- `types-aiofiles`: Type stubs for aiofiles (23.2.0.0)
- `types-Markdown`: Type stubs for Markdown (3.5.0.3)
- `types-beautifulsoup4`: Type stubs for beautifulsoup4 (4.12.0.20240311)
- `pandas-stubs`: Type stubs for pandas (2.2.0.240218)
- `types-openpyxl`: Type stubs for openpyxl (3.1.0.20240311)
- `lxml-stubs`: Type stubs for lxml (0.5.1)
- `types-python-dateutil`: Type stubs for python-dateutil (2.8.19.20240311)

### Development Dependencies

- `pytest`: Testing framework (8.0.2)
- `pytest-asyncio`: Async test support (0.23.5)
- `black`: Code formatting (24.2.0)
- `isort`: Import sorting (5.13.2)
- `mypy`: Type checking (1.8.0)
- `ruff`: Code linting (0.3.2)

### GUI Dependencies

- `PyQt6`: Modern Qt-based GUI framework (6.6.1)
