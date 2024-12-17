"""Model management functionality."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import shutil
from llama_recipes import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoConfig

class ModelManager:
    """Manages model downloading and organization."""
    
    def __init__(self):
        """Initialize model manager."""
        self.workspace_root = Path.cwd()
        self.models_dir = self.workspace_root / "data/models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.workspace_root / ".config/model_config.yaml"
        self.load_config()
        
    def load_config(self) -> None:
        """Load model configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                config = yaml.safe_load(f)
                self.config = config.get("models", {})
        else:
            raise FileNotFoundError(f"Model configuration file not found: {self.config_file}")
            
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information.
        
        Args:
            model_name: Name of model
            
        Returns:
            Dictionary of model information
        """
        if model_name not in self.config:
            raise ValueError(f"Unknown model: {model_name}")
        return self.config[model_name]
        
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models.
        
        Returns:
            Dictionary mapping model types to available versions
        """
        available = {}
        for model_name, model_info in self.config.items():
            model_dir = self.models_dir / model_name
            if model_dir.exists():
                versions = []
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir() and self.validate_model_files(model_name, version_dir.name):
                        versions.append(version_dir.name)
                available[model_name] = versions
        return available
        
    def check_model_requirements(self, model_name: str, version: str) -> bool:
        """Check if system meets model requirements.
        
        Args:
            model_name: Name of model
            version: Model version
            
        Returns:
            True if requirements are met
        """
        model_info = self.get_model_info(model_name)
        
        # Check GPU requirements
        if model_info.get("requires_gpu", False):
            import torch
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                return False
                
            # Check GPU memory
            if "min_gpu_memory" in model_info:
                required_memory = model_info["min_gpu_memory"].get(version)
                if required_memory:
                    if torch.cuda.is_available():
                        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                        if available_memory < required_memory:
                            return False
                            
        # Check API key requirements
        if model_info.get("requires_key", False):
            env_vars = model_info.get("env_vars", [])
            if not all(os.getenv(var) for var in env_vars):
                return False
                
        return True
        
    def download_model(self, model_name: str, version: str) -> bool:
        """Download a model.
        
        Args:
            model_name: Name of model to download
            version: Version of model to download
            
        Returns:
            True if download successful
        """
        try:
            if not self.check_model_requirements(model_name, version):
                raise RuntimeError("System does not meet model requirements")
                
            model_path = self.get_model_path(model_name, version)
            model_path.mkdir(parents=True, exist_ok=True)
            
            model_info = self.get_model_info(model_name)
            
            if model_name == "llama3.3":
                # Download model and tokenizer
                model_id = f"meta-llama/Llama-2-{version}"
                if "variants" in model_info:
                    variant = os.getenv("LLAMA_MODEL_VARIANT", "base")
                    if variant in model_info["variants"]:
                        model_id += f"-{variant}"
                        
                tokenizer = LlamaTokenizer.from_pretrained(model_id)
                model = LlamaForCausalLM.from_pretrained(model_id)
                
                # Configure H2O settings if enabled
                if model_info.get("h2o_config", {}).get("enabled", False):
                    h2o_config = model_info["h2o_config"]
                    model.config.window_length = h2o_config.get("window_length", 512)
                    model.config.heavy_hitter_tokens = h2o_config.get("heavy_hitter_tokens", 128)
                
                # Save locally
                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
                
            elif model_info.get("requires_key", False):
                # For API models, just save the config
                config = {
                    "name": model_info["name"],
                    "version": version,
                    "api_required": True
                }
                if "max_tokens" in model_info:
                    config["max_tokens"] = model_info["max_tokens"]
                if "model_variants" in model_info:
                    config["variants"] = model_info["model_variants"]
                    
                config_path = model_path / "config.json"
                with open(config_path, "w") as f:
                    yaml.dump(config, f)
                    
            return True
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
            
    def remove_model(self, model_name: str, version: str) -> bool:
        """Remove a downloaded model.
        
        Args:
            model_name: Name of model to remove
            version: Version of model to remove
            
        Returns:
            True if removal successful
        """
        try:
            model_path = self.get_model_path(model_name, version)
            if model_path.exists():
                shutil.rmtree(model_path)
            return True
        except Exception as e:
            print(f"Error removing model: {e}")
            return False
            
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """Get path for model files.
        
        Args:
            model_name: Type of model
            version: Optional model version
            
        Returns:
            Path to model directory
        """
        model_info = self.get_model_info(model_name)
        if version is None:
            version = model_info["versions"][0]  # Use first version as default
        return self.models_dir / model_name / version
        
    def validate_model_files(self, model_name: str, version: str) -> bool:
        """Check if model files exist and are valid.
        
        Args:
            model_name: Type of model
            version: Model version
            
        Returns:
            True if model files are valid
        """
        model_path = self.get_model_path(model_name, version)
        if not model_path.exists():
            return False
            
        model_info = self.get_model_info(model_name)
        
        # API models just need config.json
        if model_info.get("requires_key", False):
            return (model_path / "config.json").exists()
            
        # Local models need full model files
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        return all((model_path / file).exists() for file in required_files)