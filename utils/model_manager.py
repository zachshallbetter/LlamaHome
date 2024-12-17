"""Model management utilities."""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class ModelConfig:
    """Model configuration."""

    DEFAULT_CONFIG = {
        "llama": {
            "formats": ["gguf"],
            "versions": ["3.3-7b", "3.3-13b", "3.3-70b"],
            "default_version": "3.3-7b",
            "min_gpu_memory": {
                "3.3-7b": 8,
                "3.3-13b": 16,
                "3.3-70b": 40
            },
            "urls": {
                "3.3-7b": "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf",
                "3.3-13b": "https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_M.gguf",
                "3.3-70b": "https://huggingface.co/TheBloke/Llama-2-70B-GGUF/resolve/main/llama-2-70b.Q4_K_M.gguf"
            }
        },
        "gpt4": {
            "formats": ["api"],
            "versions": ["4-turbo", "4"],
            "default_version": "4-turbo",
            "api_required": True,
            "urls": {}  # No URLs needed for API models
        },
        "claude": {
            "formats": ["api"],
            "versions": ["3-opus", "3-sonnet", "3-haiku"],
            "default_version": "3-opus",
            "api_required": True,
            "urls": {}  # No URLs needed for API models
        }
    }

    def __init__(self):
        """Initialize model configuration."""
        self.config_file = Path(".config/models.json")
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_config()

    def load_config(self):
        """Load model configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config = json.load(f)
                self.model_configs = config
                self.model_types = list(config.keys())
            else:
                self.model_configs = self.DEFAULT_CONFIG
                self.model_types = list(self.DEFAULT_CONFIG.keys())
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            self.model_configs = self.DEFAULT_CONFIG
            self.model_types = list(self.DEFAULT_CONFIG.keys())

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.model_configs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model config: {e}")

    def add_model_version(self, model_type: str, version: str, gpu_memory: Optional[int] = None) -> bool:
        """Add a new model version to configuration.
        
        Args:
            model_type: Type of model
            version: Version to add
            gpu_memory: Optional GPU memory requirement in GB
            
        Returns:
            True if version was added successfully
        """
        try:
            if model_type not in self.model_types:
                return False
                
            if version not in self.model_configs[model_type]["versions"]:
                self.model_configs[model_type]["versions"].append(version)
                
                if gpu_memory and "min_gpu_memory" in self.model_configs[model_type]:
                    self.model_configs[model_type]["min_gpu_memory"][version] = gpu_memory
                    
                self.save_config()
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error adding model version: {e}")
            return False

    def remove_model_version(self, model_type: str, version: str) -> bool:
        """Remove a model version from configuration.
        
        Args:
            model_type: Type of model
            version: Version to remove
            
        Returns:
            True if version was removed successfully
        """
        try:
            if model_type not in self.model_types:
                return False
                
            if version in self.model_configs[model_type]["versions"]:
                self.model_configs[model_type]["versions"].remove(version)
                
                # Clean up all version-specific data
                if "min_gpu_memory" in self.model_configs[model_type]:
                    self.model_configs[model_type]["min_gpu_memory"].pop(version, None)
                if "urls" in self.model_configs[model_type]:
                    self.model_configs[model_type]["urls"].pop(version, None)
                    
                self.save_config()
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error removing model version: {e}")
            return False

    def update_model_version(
        self, 
        model_type: str, 
        version: str, 
        gpu_memory: Optional[int] = None,
        url: Optional[str] = None,
        make_default: bool = False
    ) -> bool:
        """Update properties of an existing model version.
        
        Args:
            model_type: Type of model
            version: Version to update
            gpu_memory: Optional new GPU memory requirement
            url: Optional new download URL
            make_default: Whether to make this version the default
            
        Returns:
            True if version was updated successfully
        """
        try:
            if model_type not in self.model_types:
                return False
                
            if version not in self.model_configs[model_type]["versions"]:
                return False
                
            # Update GPU memory if provided
            if gpu_memory is not None and "min_gpu_memory" in self.model_configs[model_type]:
                self.model_configs[model_type]["min_gpu_memory"][version] = gpu_memory
                
            # Update URL if provided
            if url is not None:
                if "urls" not in self.model_configs[model_type]:
                    self.model_configs[model_type]["urls"] = {}
                self.model_configs[model_type]["urls"][version] = url
                
            # Update default version if requested
            if make_default:
                self.model_configs[model_type]["default_version"] = version
                
            self.save_config()
            return True
            
        except Exception as e:
            logger.error(f"Error updating model version: {e}")
            return False

    def add_model_url(self, model_type: str, version: str, url: str) -> bool:
        """Add a new model URL to configuration.
        
        Args:
            model_type: Type of model
            version: Model version
            url: Download URL
            
        Returns:
            True if URL was added successfully
        """
        try:
            if model_type not in self.model_types:
                return False
                
            if "urls" not in self.model_configs[model_type]:
                self.model_configs[model_type]["urls"] = {}
                
            self.model_configs[model_type]["urls"][version] = url
            self.save_config()
            return True
        except Exception as e:
            logger.error(f"Error adding model URL: {e}")
            return False

    def get_model_url(self, model_type: str, version: str) -> Optional[str]:
        """Get download URL for a model version.
        
        Args:
            model_type: Type of model
            version: Model version
            
        Returns:
            Download URL if found, None otherwise
        """
        try:
            return self.model_configs.get(model_type, {}).get("urls", {}).get(version)
        except Exception as e:
            logger.error(f"Error getting model URL: {e}")
            return None


class ModelManager:
    """Manages model files and configurations."""

    _instance = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize model manager."""
        if self._initialized:
            return
            
        self.config = ModelConfig()
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model directories quietly
        for model_type in self.config.model_types:
            model_dir = self.models_dir / model_type
            model_dir.mkdir(exist_ok=True)
            
        self._initialized = True

    def get_model_path(self, model_type: str, version: Optional[str] = None) -> Path:
        """Get path for model files.

        Args:
            model_type: Type of model
            version: Optional model version

        Returns:
            Path to model directory
        """
        if version is None:
            version = self.config.model_configs[model_type]["default_version"]
            
        return self.models_dir / model_type / version

    def validate_model_files(self, model_type: str, version: Optional[str] = None) -> bool:
        """Check if model files exist and are valid.

        Args:
            model_type: Type of model
            version: Optional model version

        Returns:
            True if model files are valid
        """
        model_path = self.get_model_path(model_type, version)
        
        if not model_path.exists():
            return False
            
        # For GGUF models, check for any .gguf file
        if model_type == "llama":
            gguf_files = list(model_path.glob("*.gguf"))
            return len(gguf_files) > 0
            
        # For API models, check for config file with API key
        elif self.config.model_configs[model_type].get("api_required"):
            config_file = model_path / "config.json"
            if not config_file.exists():
                return False
            try:
                config = json.loads(config_file.read_text())
                return "api_key" in config
            except:
                return False
                
        return True

    def get_model_file(self, model_type: str, version: Optional[str] = None) -> Optional[Path]:
        """Get the path to the model file.

        Args:
            model_type: Type of model
            version: Optional model version

        Returns:
            Path to model file if found, None otherwise
        """
        model_path = self.get_model_path(model_type, version)
        
        if model_type == "llama":
            gguf_files = list(model_path.glob("*.gguf"))
            return gguf_files[0] if gguf_files else None
            
        return None

    def cleanup_model_files(self, model_type: str, version: Optional[str] = None) -> None:
        """Remove model files.

        Args:
            model_type: Type of model to remove
            version: Optional specific version to remove
        """
        if version:
            # Remove specific version
            model_path = self.get_model_path(model_type, version)
            if model_path.exists():
                shutil.rmtree(model_path)
        else:
            # Remove all versions
            model_path = self.models_dir / model_type
            if model_path.exists():
                shutil.rmtree(model_path)
                model_path.mkdir()  # Recreate empty directory
                
        logger.info(f"Cleaned {'all' if not version else version} model files for {model_type}")

    def list_available_models(self) -> Dict[str, List[str]]:
        """List available model versions.

        Returns:
            Dictionary mapping model types to lists of available versions
        """
        available = {}
        for model_type in self.config.model_types:
            model_dir = self.models_dir / model_type
            if model_dir.exists():
                versions = []
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir() and self.validate_model_files(model_type, version_dir.name):
                        versions.append(version_dir.name)
                available[model_type] = versions
            else:
                available[model_type] = []
                
        return available
