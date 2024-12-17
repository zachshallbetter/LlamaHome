"""
Environment setup and configuration utilities.
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Any

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def setup_environment() -> Dict[str, Any]:
    """Set up application environment."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("Setting up environment...", total=4)
            
            # Set up environment variables
            _setup_env_variables()
            progress.advance(task)
            
            # Set up configuration
            _setup_config()
            progress.advance(task)
            
            # Set up logging
            _setup_logging()
            progress.advance(task)
            
            # Set up compute environment
            _setup_compute_env()
            progress.advance(task)
            
        return {
            "success": True,
            "message": "Environment setup complete"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def _setup_env_variables():
    """Set up environment variables."""
    # Suppress unnecessary warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["ACCELERATE_LOGGING_LEVEL"] = "error"
    
    # Set up path variables if not set
    workspace_root = Path.cwd()
    
    env_vars = {
        "LLAMAHOME_ROOT": workspace_root,
        "LLAMAHOME_DATA": workspace_root / "data",
        "LLAMAHOME_MODELS": workspace_root / "models",
        "LLAMAHOME_CONFIG": workspace_root / ".config",
        "LLAMAHOME_CACHE": workspace_root / ".cache",
        "LLAMAHOME_LOGS": workspace_root / ".logs"
    }
    
    for var_name, var_path in env_vars.items():
        if var_name not in os.environ:
            os.environ[var_name] = str(var_path)

def _setup_config():
    """Set up configuration files."""
    config_dir = Path(os.environ["LLAMAHOME_CONFIG"])
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Default configurations
    default_configs = {
        "models.json": {
            "models": {
                "llama": {
                    "name": "llama",
                    "type": "llama",
                    "version": "latest"
                }
            }
        },
        "training_config.yaml": {
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10
            },
            "data": {
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            }
        }
    }
    
    # Create default config files if they don't exist
    for config_name, default_config in default_configs.items():
        config_path = config_dir / config_name
        if not config_path.exists():
            with open(config_path, "w") as f:
                if config_name.endswith(".json"):
                    import json
                    json.dump(default_config, f, indent=2)
                else:
                    yaml.dump(default_config, f, default_flow_style=False)

def _setup_logging():
    """Set up logging configuration."""
    logs_dir = Path(os.environ["LLAMAHOME_LOGS"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log files if they don't exist
    log_files = ["app.log", "error.log", "debug.log"]
    for log_file in log_files:
        log_path = logs_dir / log_file
        if not log_path.exists():
            log_path.touch()

def _setup_compute_env():
    """Set up compute environment."""
    import torch
    
    # Detect compute capabilities
    compute_info = {
        "device": "cpu",
        "backend": "cpu",
        "capabilities": []
    }
    
    # Check CUDA
    if torch.cuda.is_available():
        compute_info["device"] = "cuda"
        compute_info["backend"] = "cuda"
        compute_info["capabilities"].extend([
            f"cuda_{i}" for i in range(torch.cuda.device_count())
        ])
    
    # Check MPS (Apple Silicon)
    elif (platform.system() == "Darwin" and 
          platform.machine() == "arm64" and
          hasattr(torch.backends, "mps") and
          torch.backends.mps.is_available()):
        compute_info["device"] = "mps"
        compute_info["backend"] = "mps"
        compute_info["capabilities"].append("mps")
    
    # Save compute info
    config_dir = Path(os.environ["LLAMAHOME_CONFIG"])
    compute_config_path = config_dir / "compute_config.json"
    
    import json
    with open(compute_config_path, "w") as f:
        json.dump(compute_info, f, indent=2)

if __name__ == "__main__":
    result = setup_environment()
    if result["success"]:
        console.print("[green]Environment setup complete![/green]")
    else:
        console.print(f"[red]Environment setup failed: {result['error']}[/red]")
        sys.exit(1)
