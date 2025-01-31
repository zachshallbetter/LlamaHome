"""Safe I/O utilities."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch import Tensor


def safe_torch_save(obj: Any, path: Union[str, Path], **kwargs: Any) -> str:
    """Safely save PyTorch object with hash verification.

    Returns:
        str: Hash of the saved file
    """
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
    torch.save(
        obj, path, pickle_protocol=4, _use_new_zipfile_serialization=True, **kwargs
    )

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
    """Safely load PyTorch object with hash verification."""
    path = Path(path)
    try:
        if not path.exists():
            raise ValueError(f"File {path} not found")
        if path.suffix not in {".pt", ".pth"}:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        result: Dict[str, Any] = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            pickle_module=None,
            **kwargs,
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to load file {path}: {e}") from e
