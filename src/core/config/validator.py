from pathlib import Path
from typing import Dict, List

import toml
from pydantic import BaseModel


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    def __init__(self, schema_dir: Path = Path("config/schemas")):
        self.schema_dir = schema_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

    async def validate_config(self, config: Dict, config_type: str) -> ValidationResult:
        """Validate configuration against schema."""
        try:
            # Load schema
            schema_path = self.schema_dir / f"{config_type}.toml"
            schema = toml.load(schema_path)

            # Validate structure
            self._validate_structure(config, schema)

            # Validate types
            self._validate_types(config, schema)

            # Validate dependencies
            self._validate_dependencies(config)

            # Validate constraints
            self._validate_constraints(config)

            return ValidationResult(
                is_valid=len(self.errors) == 0,
                errors=self.errors,
                warnings=self.warnings,
            )

        except Exception as e:
            self.errors.append(f"Validation error: {str(e)}")
            return ValidationResult(
                is_valid=False, errors=self.errors, warnings=self.warnings
            )

    def _validate_structure(self, config: Dict, schema: Dict) -> None:
        """Validate configuration structure against schema."""
        for key, value in schema.items():
            if key not in config:
                self.errors.append(f"Missing required key: {key}")
            elif isinstance(value, dict):
                if not isinstance(config[key], dict):
                    self.errors.append(f"Invalid type for {key}: expected dict")
                else:
                    self._validate_structure(config[key], value)

    def _validate_types(self, config: Dict, schema: Dict) -> None:
        """Validate configuration value types."""
        for key, value in config.items():
            if key in schema:
                expected_type = schema[key].get("type")
                if expected_type and not isinstance(value, eval(expected_type)):
                    self.errors.append(
                        f"Invalid type for {key}: expected {expected_type}"
                    )

    def _validate_dependencies(self, config: Dict) -> None:
        """Validate configuration dependencies."""
        if "models" in config:
            for model in config["models"].values():
                if model.get("requires_gpu", False):
                    if "gpu_memory" not in model:
                        self.errors.append("Missing gpu_memory config for GPU model")

    def _validate_constraints(self, config: Dict) -> None:
        """Validate configuration constraints."""
        if "resources" in config:
            resources = config["resources"]
            if "gpu_memory_fraction" in resources:
                fraction = resources["gpu_memory_fraction"]
                if not (0 < fraction <= 1):
                    self.errors.append("gpu_memory_fraction must be between 0 and 1")
