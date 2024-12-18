"""Document format conversion utilities."""

import json
from pathlib import Path
from typing import Dict, Optional, Union



try:
    import fitz


    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from ...core.utils import LogManager, LogTemplates



logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class DocumentConverter:
    """Converts documents between different formats."""


    def __init__(self):
        """Initialize document converter."""
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available, PDF conversion will be disabled")


    def convert_pdf_to_text(self, pdf_path: Union[str, Path]) -> Optional[str]:
        """Convert PDF to plain text.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or None if conversion fails
        """
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF not available for PDF conversion")
            return None

        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None

            doc = fitz.open(str(pdf_path))
            text = ""

            for page in doc:
                text += page.get_text()

            return text.strip()

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return None


    def convert_text_to_json(self, text: str, structure: Optional[Dict] = None) -> Optional[Dict]:
        """Convert text to JSON format.

        Args:
            text: Input text
            structure: Optional structure definition

        Returns:
            JSON dict or None if conversion fails
        """
        try:
            if structure:
                # Convert according to structure
                result = {}
                for key, format_spec in structure.items():
                    if format_spec == 'string':
                        result[key] = text
                    elif format_spec == 'lines':
                        result[key] = text.splitlines()
                    elif format_spec == 'words':
                        result[key] = text.split()
                return result
            else:
                # Simple conversion
                return {
                    'text': text,
                    'lines': text.splitlines(),
                    'words': text.split()
                }

        except Exception as e:
            logger.error(f"JSON conversion failed: {e}")
            return None


    def convert_json_to_text(self, data: Dict, format_spec: Optional[str] = None) -> Optional[str]:
        """Convert JSON to text format.

        Args:
            data: Input JSON data
            format_spec: Optional format specification

        Returns:
            Formatted text or None if conversion fails
        """
        try:
            if format_spec == 'pretty':
                return json.dumps(data, indent=2)
            elif format_spec == 'compact':
                return json.dumps(data, separators=(',', ':'))
            elif format_spec == 'lines':
                return '\n'.join(str(item) for item in data.values())
            else:
                return str(data)

        except Exception as e:
            logger.error(f"Text conversion failed: {e}")
            return None
