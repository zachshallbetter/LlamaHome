"""Configuration handling for LlamaHome."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import toml
from dotenv import load_dotenv
from pydantic import BaseModel

from ..utils import LogManager, LogTemplates
from .base import (
    BaseConfig,
    CacheConfig,
    ConfigError,
    MonitoringConfig,
    OptimizationConfig,
    ProcessingConfig,
    ResourceConfig,
)
from .constants import CACHE_DIR, DATA_DIR, LOCAL_CONFIG_DIR, ROOT_DIR

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

T = TypeVar("T", bound=BaseConfig)


@dataclass
class ConfigValidationError:
    """Configuration validation error."""

    path: str
    message: str
    value: Any = None


@dataclass
class ConfigData:
    """Configuration data container."""

    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    code_check: Dict[str, Any]
    type_check: Dict[str, Any]
    errors: List[ConfigValidationError] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigData":
        """Create ConfigData from dictionary.

        Args:
            data: Raw configuration dictionary

        Returns:
            ConfigData instance
        """
        return cls(
            model_config=data.get("model_config", {}),
            training_config=data.get("training_config", {}),
            code_check=data.get("code_check", {}),
            type_check=data.get("type_check", {}),
        )


class ConfigManager:
    """Manages configuration loading and validation across the application."""

    REQUIRED_ENV_VARS = {
        "LLAMAHOME_ENV",
        "LLAMA_MODEL",
        "LLAMA_MODEL_SIZE",
        "LLAMA_MODEL_VARIANT",
        "LLAMAHOME_LOG_LEVEL",
    }

    def __init__(
        self, config_dir: Path = Path("config"), env_prefix: str = "LLAMAHOME_"
    ):
        self.config_dir = config_dir
        self.env_prefix = env_prefix
        self.configs: dict[str, BaseConfig] = {}
        self.workspace_root = ROOT_DIR
        self.local_config_dir = LOCAL_CONFIG_DIR
        self._config: Optional[ConfigData] = None
        self._load_environment()

    def _load_environment(self) -> None:
        """Load environment variables."""
        load_dotenv()
        missing_vars = self.REQUIRED_ENV_VARS - set(os.environ.keys())
        if missing_vars:
            logger.warning(f"Missing required environment variables: {missing_vars}")

    def _get_config_path(self, filename: str) -> Path:
        """Get configuration file path, checking local config first.

        Args:
            filename: Name of the configuration file

        Returns:
            Path to the configuration file
        """
        local_path = self.local_config_dir / filename
        if local_path.exists():
            return local_path
        return self.config_dir / filename

    async def load_config(
        self, config_type: type[T], name: str, file_name: str | None = None
    ) -> T:
        """Load configuration of specified type."""
        try:
            # Try loading from file first
            if file_name:
                config_path = self.config_dir / file_name
                if config_path.exists():
                    config = config_type.load_from_file(config_path)
                    self.configs[name] = config
                    return config

            # Fall back to environment variables
            config = config_type.load_from_env(self.env_prefix)
            self.configs[name] = config
            return config

        except Exception as e:
            raise ConfigError(f"Failed to load {name} config: {str(e)}")

    async def save_config(self, name: str, file_name: str | None = None) -> None:
        """Save configuration to file."""
        if name not in self.configs:
            raise ConfigError(f"Config {name} not found")

        if file_name:
            config_path = self.config_dir / file_name
            self.configs[name].save_to_file(config_path)

    async def get_config(self, name: str) -> BaseConfig:
        """Get loaded configuration by name."""
        if name not in self.configs:
            raise ConfigError(f"Config {name} not found")
        return self.configs[name]

    async def update_config(self, name: str, updates: dict[str, Any]) -> None:
        """Update configuration values."""
        if name not in self.configs:
            raise ConfigError(f"Config {name} not found")

        config = self.configs[name]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

    async def merge_configs(self, name: str, other_config: BaseConfig) -> None:
        """Merge another configuration into existing one."""
        if name not in self.configs:
            raise ConfigError(f"Config {name} not found")

        self.configs[name].merge(other_config)

    def load_config_data(self) -> ConfigData:
        """Load configuration from files.

        Returns:
            Loaded configuration data

        Raises:
            FileNotFoundError: If required config files are missing
            ValueError: If config files are invalid
        """
        if self._config:
            return self._config

        try:
            # Load model config
            model_config = self._load_json("models.json")

            # Load training config
            training_config = self._load_yaml("training_config.toml")

            # Load code check config
            code_check = self._load_yaml("code_check.toml")

            # Load type check config
            type_check = self._load_ini("llamahome.types.ini")

            self._config = ConfigData(
                model_config=model_config,
                training_config=training_config,
                code_check=code_check,
                type_check=type_check,
            )

            # Validate loaded config
            self.validate()

            return self._config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load and parse JSON config file.

        Args:
            filename: Name of config file to load

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        file_path = self._get_config_path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load and parse YAML config file.

        Args:
            filename: Name of config file to load

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        file_path = self._get_config_path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return toml.load(f) or {}
        except toml.TomlDecodeError as e:
            raise ValueError(f"Invalid YAML in {filename}: {e}")

    def _load_ini(self, filename: str) -> Dict[str, Any]:
        """Load and parse INI config file.

        Args:
            filename: Name of config file to load

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        file_path = self._get_config_path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            import configparser

            parser = configparser.ConfigParser()
            parser.read(file_path)
            return {section: dict(parser[section]) for section in parser.sections()}
        except configparser.Error as e:
            raise ValueError(f"Invalid INI in {filename}: {e}")

    def validate(self) -> bool:
        """Validate current configuration.

        Returns:
            True if configuration is valid
        """
        if not self._config:
            return False

        self._config.errors.clear()
        valid = True

        # Validate model configuration
        valid &= self._validate_model_config()

        # Validate training configuration
        valid &= self._validate_training_config()

        # Validate paths
        valid &= self._validate_paths()

        return valid

    def _validate_model_config(self) -> bool:
        """Validate model configuration.

        Returns:
            True if model configuration is valid
        """
        valid = True
        config = self._config.model_config

        # Check required fields
        for model_type, model_info in config.items():
            if "versions" not in model_info:
                self._add_error("model_config", f"Missing versions for {model_type}")
                valid = False

            if "formats" not in model_info:
                self._add_error("model_config", f"Missing formats for {model_type}")
                valid = False

            if "default_version" in model_info:
                if model_info["default_version"] not in model_info.get("versions", []):
                    self._add_error(
                        "model_config", f"Invalid default version for {model_type}"
                    )
                    valid = False

        return valid

    def _validate_training_config(self) -> bool:
        """Validate training configuration.

        Returns:
            True if training configuration is valid
        """
        valid = True
        config = self._config.training_config

        # Validate batch size
        batch_size = config.get("batch_size")
        if not isinstance(batch_size, int) or batch_size < 1:
            self._add_error("training_config", "Invalid batch size")
            valid = False

        # Validate learning rate
        lr = config.get("learning_rate")
        if not isinstance(lr, float) or lr <= 0:
            self._add_error("training_config", "Invalid learning rate")
            valid = False

        # Validate LoRA config if present
        if "lora" in config:
            lora = config["lora"]
            if not isinstance(lora.get("r"), int) or lora.get("r") < 1:
                self._add_error("training_config", "Invalid LoRA rank")
                valid = False

        return valid

    def _validate_paths(self) -> bool:
        """Validate configured paths.

        Returns:
            True if all paths are valid
        """
        valid = True

        # Check data directories
        if not DATA_DIR.exists():
            self._add_error("paths", "Data directory does not exist")
            valid = False

        # Check cache directory
        if not CACHE_DIR.exists():
            self._add_error("paths", "Cache directory does not exist")
            valid = False

        # Check log directory
        log_dir = ROOT_DIR / ".logs"
        if not log_dir.exists():
            self._add_error("paths", "Log directory does not exist")
            valid = False

        return valid

    def _add_error(self, path: str, message: str, value: Any = None) -> None:
        """Add validation error.

        Args:
            path: Configuration path
            message: Error message
            value: Optional invalid value
        """
        if self._config:
            self._config.errors.append(ConfigValidationError(path, message, value))

    def update_training_config(self, updates: Dict[str, Any]) -> bool:
        """Update training configuration.

        Args:
            updates: Configuration updates

        Returns:
            True if update successful
        """
        if not self._config:
            return False

        # Deep update training config
        self._deep_update(self._config.training_config, updates)

        # Validate after update
        if not self._validate_training_config():
            return False

        # Save updated config
        return self._save_yaml("training_config.toml", self._config.training_config)

    def update_model_config(self, model_type: str, updates: Dict[str, Any]) -> bool:
        """Update model configuration.

        Args:
            model_type: Type of model to update
            updates: Configuration updates

        Returns:
            True if update successful
        """
        if not self._config or model_type not in self._config.model_config:
            return False

        # Deep update model config
        self._deep_update(self._config.model_config[model_type], updates)

        # Validate after update
        if not self._validate_model_config():
            return False

        # Save updated config
        return self._save_json("models.json", self._config.model_config)

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update dictionary.

        Args:
            d: Dictionary to update
            u: Update dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def _save_json(self, filename: str, data: Dict[str, Any]) -> bool:
        """Save configuration to JSON file.

        Args:
            filename: Name of file to save
            data: Data to save

        Returns:
            True if save successful
        """
        try:
            file_path = self._get_config_path(filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON config: {e}")
            return False

    def _save_yaml(self, filename: str, data: Dict[str, Any]) -> bool:
        """Save configuration to YAML file.

        Args:
            filename: Name of file to save
            data: Data to save

        Returns:
            True if save successful
        """
        try:
            file_path = self._get_config_path(filename)
            with open(file_path, "w", encoding="utf-8") as f:
                toml.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save YAML config: {e}")
            return False

    def get_errors(self) -> List[ConfigValidationError]:
        """Get current validation errors.

        Returns:
            List of validation errors
        """
        return self._config.errors if self._config else []


class ApplicationConfig(BaseModel):
    """Complete application configuration."""

    resources: ResourceConfig
    processing: ProcessingConfig
    optimization: OptimizationConfig
    monitoring: MonitoringConfig
    cache: CacheConfig

    @classmethod
    async def load(
        cls, config_dir: Path = Path("config"), env_prefix: str = "LLAMAHOME_"
    ) -> "ApplicationConfig":
        """Load complete application configuration."""
        manager = ConfigManager(config_dir, env_prefix)

        resources = await manager.load_config(
            ResourceConfig, "resources", "resource_config.toml"
        )
        processing = await manager.load_config(
            ProcessingConfig, "processing", "processing_config.toml"
        )
        optimization = await manager.load_config(
            OptimizationConfig, "optimization", "optimization_config.toml"
        )
        monitoring = await manager.load_config(
            MonitoringConfig, "monitoring", "monitoring_config.toml"
        )
        cache = await manager.load_config(CacheConfig, "cache", "cache_config.toml")

        return cls(
            resources=resources,
            processing=processing,
            optimization=optimization,
            monitoring=monitoring,
            cache=cache,
        )
