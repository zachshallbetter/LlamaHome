"""Format converter implementation."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class FormatConverter:
    """Handles conversion between different data formats."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize converter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.supported_formats = {".csv", ".json", ".txt", ".pdf"}

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """Process a single file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            
        Returns:
            bool: True if conversion successful
            
        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If input file not found
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        try:
            if input_path.suffix == ".csv":
                return self._process_csv(input_path, output_path)
            elif input_path.suffix == ".json":
                return self._process_json(input_path, output_path)
            elif input_path.suffix == ".txt":
                return self._process_text(input_path, output_path)
            elif input_path.suffix == ".pdf":
                return self._process_pdf(input_path, output_path)
            return False
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            return False

    def process_directory(
        self, input_dir: Path, output_dir: Path
    ) -> Dict[str, List[Path]]:
        """Process all files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            
        Returns:
            Dict containing lists of successful and failed files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {"successful": [], "failed": []}

        for input_file in input_dir.glob("*"):
            if input_file.suffix in self.supported_formats:
                output_file = output_dir / f"{input_file.stem}.jsonl"
                if self.process_file(input_file, output_file):
                    results["successful"].append(input_file)
                else:
                    results["failed"].append(input_file)

        return results

    def _process_csv(self, input_path: Path, output_path: Path) -> bool:
        """Process CSV file to JSONL.
        
        Args:
            input_path: Input CSV file path
            output_path: Output JSONL file path
            
        Returns:
            bool: True if successful
        """
        try:
            with open(input_path, newline="") as f_in:
                reader = csv.DictReader(f_in)
                with open(output_path, "w") as f_out:
                    for row in reader:
                        json.dump(row, f_out)
                        f_out.write("\n")
            return True
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            return False

    def _process_json(self, input_path: Path, output_path: Path) -> bool:
        """Process JSON file to JSONL.
        
        Args:
            input_path: Input JSON file path
            output_path: Output JSONL file path
            
        Returns:
            bool: True if successful
        """
        try:
            with open(input_path) as f_in:
                data = json.load(f_in)
                with open(output_path, "w") as f_out:
                    if isinstance(data, list):
                        for item in data:
                            json.dump(item, f_out)
                            f_out.write("\n")
                    else:
                        json.dump(data, f_out)
                        f_out.write("\n")
            return True
        except Exception as e:
            logger.error(f"Error processing JSON file: {e}")
            return False

    def _process_text(self, input_path: Path, output_path: Path) -> bool:
        """Process text file to JSONL.
        
        Args:
            input_path: Input text file path
            output_path: Output JSONL file path
            
        Returns:
            bool: True if successful
        """
        try:
            with open(input_path) as f_in:
                with open(output_path, "w") as f_out:
                    for line in f_in:
                        json.dump({"text": line.strip()}, f_out)
                        f_out.write("\n")
            return True
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            return False

    def _process_pdf(self, input_path: Path, output_path: Path) -> bool:
        """Process PDF file to JSONL.
        
        Args:
            input_path: Input PDF file path
            output_path: Output JSONL file path
            
        Returns:
            bool: True if successful
        """
        try:
            import fitz  # Import here to avoid dependency if not needed

            doc = fitz.open(input_path)
            with open(output_path, "w") as f_out:
                for page in doc:
                    text = page.get_text()
                    json.dump({"page": page.number + 1, "text": text}, f_out)
                    f_out.write("\n")
            return True
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            return False


def create_converter(config: Optional[Dict[str, Any]] = None) -> FormatConverter:
    """Create a format converter instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FormatConverter instance
    """
    return FormatConverter(config) 