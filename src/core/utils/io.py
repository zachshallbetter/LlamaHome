"""Safe I/O utilities."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch import Tensor


def safe_torch_save(obj: Any, path: Union[str, Path], **kwargs: Any) -> str:
    """Safely save PyTorch object with hash verification."""
    path = Path(path)

    # Convert to JSON-safe format first
    if isinstance(obj, dict):
        obj = {
            str(k): (v.tolist() if isinstance(v, Tensor) else v) for k, v in obj.items()
        }

    # Save as JSON first for safety
    json_path = path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

    # Then save PyTorch format with safe_storage=True
    torch.save(obj, path, pickle_protocol=4, _use_new_zipfile_serialization=True, **kwargs)

    # Calculate hash of both files
    with open(path, "rb") as f:
        torch_hash = hashlib.sha256(f.read()).hexdigest()

    with open(json_path, "rb") as f:
        json_hash = hashlib.sha256(f.read()).hexdigest()

    # Save hashes
    hash_path = path.with_suffix(".hash")
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump({"torch": torch_hash, "json": json_hash}, f)

    return torch_hash


def safe_torch_load(path: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
    """Safely load PyTorch object with hash verification.
    
    Args:
        path: Path to the saved PyTorch file
        **kwargs: Additional arguments passed to torch.load
        
    Returns:
        Dict containing the loaded data
        
    Raises:
        ValueError: If files are missing or hash verification fails
    """
    path = Path(path)
    json_path = path.with_suffix(".json")
    hash_path = path.with_suffix(".hash")

    if not all(p.exists() for p in (path, json_path, hash_path)):
        raise ValueError("Missing required files")

    # Verify hashes
    with open(hash_path, encoding="utf-8") as f:
        hashes = json.load(f)

    with open(path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != hashes["torch"]:
            raise ValueError("PyTorch file hash verification failed")

    with open(json_path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != hashes["json"]:
            raise ValueError("JSON file hash verification failed")

    # Load both formats and verify
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)

    # Use safe loading options
    torch_data = torch.load(path, map_location="cpu", weights_only=True, **kwargs)

    if json_data.keys() != torch_data.keys():
        raise ValueError("Data format mismatch between JSON and PyTorch files")

    return torch_data
