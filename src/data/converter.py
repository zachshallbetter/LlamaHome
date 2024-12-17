"""Format conversion module for transforming various file formats to JSONL."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import fitz  # PyMuPDF for PDF processing

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

class FormatConverter:
    """Handles conversion of various input formats to JSONL."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the converter with optional configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.supported_formats = {
            ".csv": self._process_csv,
            ".json": self._process_json,
            ".txt": self._process_txt,
            ".pdf": self._process_pdf,
        }
        logger.debug("Initialized FormatConverter with supported formats: %s", 
                    list(self.supported_formats.keys()))

    def process_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """Process a single file and convert it to JSONL format.

        Args:
            input_path: Path to input file
            output_path: Path to save JSONL output

        Returns:
            bool: True if processing successful

        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If input file doesn't exist
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            raise FileNotFoundError(f"Input file not found: {input_path}")

        suffix = input_path.suffix.lower()
        if suffix not in self.supported_formats:
            logger.error("Unsupported file format: %s", suffix)
            raise ValueError(f"Unsupported file format: {suffix}")

        try:
            logger.info("Starting processing of %s to %s", input_path, output_path)
            processor = self.supported_formats[suffix]
            result = processor(input_path, output_path)
            logger.info("Successfully processed %s to %s", input_path, output_path)
            return result
        except Exception as e:
            logger.error("Error processing %s: %s", input_path, e)
            raise

    def _process_csv(self, input_path: Path, output_path: Path) -> bool:
        """Process CSV file to JSONL format.

        Args:
            input_path: Path to CSV file
            output_path: Path to save JSONL output

        Returns:
            bool: True if successful
        """
        try:
            logger.debug("Processing CSV file: %s", input_path)
            with (
                open(input_path, "r", encoding="utf-8") as csv_file,
                open(output_path, "w", encoding="utf-8") as jsonl_file,
            ):
                reader = csv.DictReader(csv_file)
                for row in reader:
                    json.dump(row, jsonl_file)
                    jsonl_file.write("\n")
            logger.debug("CSV processing completed: %s", input_path)
            return True
        except Exception as e:
            logger.error("CSV processing error for %s: %s", input_path, e)
            raise

    def _process_json(self, input_path: Path, output_path: Path) -> bool:
        """Process JSON file to JSONL format.

        Args:
            input_path: Path to JSON file
            output_path: Path to save JSONL output

        Returns:
            bool: True if successful
        """
        try:
            logger.debug("Processing JSON file: %s", input_path)
            with open(input_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                if isinstance(data, list):
                    for item in data:
                        json.dump(item, jsonl_file)
                        jsonl_file.write("\n")
                else:
                    json.dump(data, jsonl_file)
                    jsonl_file.write("\n")
            logger.debug("JSON processing completed: %s", input_path)
            return True
        except Exception as e:
            logger.error("JSON processing error for %s: %s", input_path, e)
            raise

    def _process_txt(self, input_path: Path, output_path: Path) -> bool:
        """Process text file to JSONL format.

        Args:
            input_path: Path to text file
            output_path: Path to save JSONL output

        Returns:
            bool: True if successful
        """
        try:
            logger.debug("Processing text file: %s", input_path)
            with (
                open(input_path, "r", encoding="utf-8") as txt_file,
                open(output_path, "w", encoding="utf-8") as jsonl_file,
            ):
                for line in txt_file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        entry = {"text": line}
                        json.dump(entry, jsonl_file)
                        jsonl_file.write("\n")
            logger.debug("Text processing completed: %s", input_path)
            return True
        except Exception as e:
            logger.error("Text processing error for %s: %s", input_path, e)
            raise

    def _process_pdf(self, input_path: Path, output_path: Path) -> bool:
        """Process PDF file to JSONL format.

        Args:
            input_path: Path to PDF file
            output_path: Path to save JSONL output

        Returns:
            bool: True if successful
        """
        try:
            logger.debug("Processing PDF file: %s", input_path)
            doc = fitz.open(input_path)
            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        entry = {"page": page_num + 1, "text": text.strip()}
                        json.dump(entry, jsonl_file)
                        jsonl_file.write("\n")
            logger.debug("PDF processing completed: %s", input_path)
            return True
        except Exception as e:
            logger.error("PDF processing error for %s: %s", input_path, e)
            raise

    def process_directory(
        self, input_dir: Union[str, Path], output_dir: Union[str, Path], recursive: bool = False
    ) -> Dict[str, List[str]]:
        """Process all supported files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            recursive: Whether to process subdirectories

        Returns:
            Dict with successful and failed files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, List[str]] = {"successful": [], "failed": []}
        logger.info("Processing directory: %s", input_dir)
        logger.debug("Output directory: %s, Recursive: %s", output_dir, recursive)

        pattern = "**/*" if recursive else "*"
        for file_path in input_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                output_path = output_dir / f"{file_path.stem}.jsonl"
                try:
                    self.process_file(file_path, output_path)
                    results["successful"].append(str(file_path))
                    logger.debug("Successfully processed: %s", file_path)
                except Exception as e:
                    logger.error("Failed to process %s: %s", file_path, e)
                    results["failed"].append(str(file_path))

        logger.info("Directory processing completed. Successful: %d, Failed: %d",
                   len(results["successful"]), len(results["failed"]))
        return results


def create_converter(config: Optional[Dict] = None) -> FormatConverter:
    """Factory function to create a FormatConverter instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        FormatConverter instance
    """
    logger.debug("Creating new FormatConverter instance with config: %s", config)
    return FormatConverter(config)
