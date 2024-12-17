"""Model setup and download utilities."""

import os
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
from typing import Optional
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

from utils.log_manager import LogManager, LogTemplates
from utils.model_manager import ModelManager

# Load environment variables
load_dotenv()

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()

class ModelSetup:
    """Handles model setup and downloads."""

    _instance = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super(ModelSetup, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize model setup."""
        if self._initialized:
            return
        self.model_manager = ModelManager()
        self._initialized = True
        
        # Set up paths
        self.workspace_root = Path.cwd()
        self.model_path = self.workspace_root / "data/models"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Load environment variables
        self.model_name = os.getenv("LLAMA_MODEL", "llama3.3")
        self.model_size = os.getenv("LLAMA_MODEL_SIZE", "13b")
        self.model_variant = os.getenv("LLAMA_MODEL_VARIANT", "chat")
        self.model_quant = os.getenv("LLAMA_MODEL_QUANT", "f16")

    def _run_command(self, command: str) -> bool:
        """Run a shell command and return success status."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command failed: {e.stderr}")
            return False
        except Exception as e:
            logger.warning(f"Error running command: {e}")
            return False

    def setup_model(
        self,
        model_type: str,
        version: Optional[str] = None,
        force_setup: bool = False
    ) -> bool:
        """Set up a model for use."""
        try:
            if version is None:
                version = self.model_manager.config.model_configs[model_type]["default_version"]

            # Get model path
            model_path = self.model_path / model_type / version
            model_path.mkdir(parents=True, exist_ok=True)

            # Check if model already exists
            if not force_setup and self.model_manager.validate_model_files(model_type, version):
                logger.info(f"Model {model_type} {version} already exists")
                return True

            # Handle different model types
            if model_type == "llama":
                success = self._setup_llama_model(version, model_path)
                if success:
                    logger.info(f"Successfully set up LLaMA {version}")
                    return True

            elif model_type in ["gpt4", "claude"]:
                # API models just need a config file
                config_file = model_path / "config.json"
                if not config_file.exists():
                    config_file.write_text("{}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to set up model: {e}")
            return False

    def _setup_llama_model(self, version: str, model_path: Path) -> bool:
        """Set up LLaMA model using CLI commands."""
        try:
            # Ensure llama-stack is installed/updated
            console.print("Checking llama-stack installation...")
            if not self._run_command("pip install -U llama-stack"):
                return False

            # List available models
            console.print("Fetching available models...")
            if not self._run_command("llama-cli models list"):
                return False

            # Construct model identifier
            model_id = f"{self.model_name}-{self.model_size}"
            if self.model_variant:
                model_id += f"-{self.model_variant}"
            if self.model_quant:
                model_id += f"-{self.model_quant}"

            # Download model
            console.print(f"Downloading model {model_id}...")
            download_cmd = f"llama-cli models download {model_id} --output-dir {model_path}"
            if not self._run_command(download_cmd):
                return False

            # Verify download
            if not any(model_path.glob("*.safetensors")):
                logger.warning("Model files not found after download")
                return False

            return True

        except Exception as e:
            logger.warning(f"Failed to set up LLaMA model: {e}")
            return False

    def list_available_models(self) -> None:
        """List all available models."""
        try:
            console.print("Available models:")
            self._run_command("llama-cli models list")
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")

    def show_model_details(self, model_id: str) -> None:
        """Show details for a specific model."""
        try:
            console.print(f"Details for model {model_id}:")
            self._run_command(f"llama-cli models show {model_id}")
        except Exception as e:
            logger.warning(f"Failed to show model details: {e}")
