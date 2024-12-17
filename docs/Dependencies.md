# LlamaHome Dependencies

## Core Dependencies

### AI and Machine Learning

- `torch`: PyTorch for deep learning operations (used in H2O implementation)
  - Version: 2.0.1
  - Requires: Python >=3.11,<3.13
  - GPU Support: CUDA 12.1 pre-built
- `transformers`: Hugging Face transformers for Llama 3.3 integration (4.47.0)
- `tokenizers`: Fast tokenization for text processing (0.21.0)
- `h2o`: Heavy-Hitter Oracle implementation (3.46.0)
- `rouge`: ROUGE score evaluation for model outputs (1.0.1)
- `psutil`: System resource monitoring for benchmarks (5.9.0)

### Document Processing

- `nltk`: Natural language processing utilities (3.8)
- `pandas`: Data manipulation and analysis (2.1.4)
- `numpy`: Numerical computing (1.26.4)
- `PyMuPDF`: PDF document processing (1.23.0)
- `python-docx`: Microsoft Word document handling (1.0.0)
- `openpyxl`: Excel file processing (3.1.0)
- `python-pptx`: PowerPoint file processing (0.6.21)
- `ebooklib`: E-book format support (0.18)
- `striprtf`: RTF document processing (0.0.28)
- `olefile`: Legacy Microsoft Office format support (0.47)
- `odfpy`: OpenDocument format support (1.4.1)
- `ezodf`: Additional OpenDocument support (0.3.2)
- `lxml`: XML and HTML processing (5.0.0)
- `beautifulsoup4`: Web content parsing (4.12.0)
- `console`: Console utilities (0.9909)

### File Handling and I/O

- `aiofiles`: Asynchronous file operations (23.2.0)
- `xopen`: Efficient compressed file handling (2.0.2)
- `markdown`: Markdown processing (3.5.0)

### CLI and Configuration

- `click`: Command line interface creation (8.1.7)
- `python-dotenv`: Environment variable management (1.0.0)
- `pyyaml`: YAML configuration handling (6.0.2)
- `rich`: Rich text and formatting for CLI (13.9.4)
- `requests`: HTTP client for downloads (2.31.0)
- `tqdm`: Progress bars (4.66.0)
- `structlog`: Structured logging (24.4.0)

### Type Stubs

- `types-requests`: Type stubs for requests (2.31.0.20240311)
- `types-tqdm`: Type stubs for tqdm (4.66.0.20240106)
- `types-PyYAML`: Type stubs for PyYAML (6.0.12.12)
- `types-aiofiles`: Type stubs for aiofiles (23.2.0.0)
- `types-Markdown`: Type stubs for Markdown (3.5.0.3)
- `types-beautifulsoup4`: Type stubs for beautifulsoup4 (4.12.0.20240229)
- `pandas-stubs`: Type stubs for pandas (2.1.1.230928)
- `types-openpyxl`: Type stubs for openpyxl (3.1.0.20240310)
- `lxml-stubs`: Type stubs for lxml (0.5.1)
- `types-python-dateutil`: Type stubs for python-dateutil (2.8.19.20240311)

### Development Dependencies

- `pytest`: Testing framework (8.0.0)
- `pytest-asyncio`: Async test support (0.23.0)
- `pytest-mock`: Mocking for tests (3.12.0)
- `black`: Code formatting (24.1.0)
- `isort`: Import sorting (5.13.0)
- `mypy`: Type checking (1.8.0)
- `flake8`: Code linting (7.0.0)
- `jupyter`: Jupyter notebooks (1.0.0)
- `notebook`: Jupyter notebook server (7.0.0)
- `autopep8`: Code formatting (2.0.4)

### GUI Dependencies

- `PyQt6`: Modern Qt-based GUI framework (6.6.1)
