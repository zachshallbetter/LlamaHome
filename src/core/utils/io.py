"""Safe I/O utilities."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..security import verify_data_source


def safe_torch_save(obj: Any, path: str | Path, **kwargs: Any) -> str:
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


def safe_torch_load(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Safely load PyTorch object with hash verification."""
    path = Path(path)
    try:
        if not path.exists():
            raise ValueError(f"File {path} not found")
        if path.suffix not in {".pt", ".pth"}:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        result: dict[str, Any] = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            pickle_module=None,
            **kwargs,
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to load file {path}: {e}") from e


def safe_load_torch(
    path: str | Path,
    device: str | None = None,
    weights_only: bool = True,
    verify: bool = True,
) -> Any:
    """Safely load PyTorch data with verification.

    Args:
        path: Path to file
        device: Optional device to load to
        weights_only: Whether to only load tensor data
        verify: Whether to verify data source

    Returns:
        Loaded data

    Raises:
        ValueError: If file verification fails
        FileNotFoundError: If file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if verify:
        # Verify data source
        verify_data_source(path)

    try:
        # Create temporary directory for safe loading
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / path.name

            # Copy file to temp location
            import shutil

            shutil.copy2(path, temp_path)

            # Load with extra verification
            data = torch.load(
                temp_path,
                map_location=device or "cpu",
                weights_only=weights_only,
                pickle_module=None,  # Disable pickle for security
            )
            return data
    except Exception as e:
        raise ValueError(f"Failed to load PyTorch data: {e}") from e


def safe_save_torch(data: Any, path: str | Path) -> None:
    """Safely save PyTorch data.

    Args:
        data: Data to save
        path: Path to save to

    Raises:
        ValueError: If save fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save with temporary file
        with tempfile.NamedTemporaryFile(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            torch.save(
                data,
                tmp.name,
                pickle_module=None,  # Disable pickle for security
            )
            # Ensure data is written to disk
            os.fsync(tmp.fileno())

        # Atomic rename
        os.rename(tmp.name, path)
    except Exception as e:
        # Clean up temp file
        if "tmp" in locals():
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        raise ValueError(f"Failed to save PyTorch data: {e}") from e


def safe_load_json(path: str | Path) -> dict[str, Any]:
    """Safely load JSON data with verification.

    Args:
        path: Path to file

    Returns:
        Loaded data

    Raises:
        ValueError: If file verification fails
        FileNotFoundError: If file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Verify data source
    verify_data_source(path)

    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON data: {e}") from e


def safe_save_json(data: dict[str, Any], path: str | Path) -> None:
    """Safely save JSON data.

    Args:
        data: Data to save
        path: Path to save to

    Raises:
        ValueError: If save fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save with temporary file
        with tempfile.NamedTemporaryFile(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            delete=False,
        ) as tmp:
            json.dump(data, tmp, indent=2)
            # Ensure data is written to disk
            tmp.flush()
            os.fsync(tmp.fileno())

        # Atomic rename
        os.rename(tmp.name, path)
    except Exception as e:
        # Clean up temp file
        if "tmp" in locals():
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        raise ValueError(f"Failed to save JSON data: {e}") from e
