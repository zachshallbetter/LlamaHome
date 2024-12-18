from typing import Dict, List, Optional, Union



try:
    from jsonschema import Draft7Validator


    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from ...core.utils import LogManager, LogTemplates
from .schema import (


    TRAINING_CONFIG_SCHEMA,
    MODEL_CONFIG_SCHEMA,
    DATA_CONFIG_SCHEMA,
    validate_schema
)

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class DataValidator:
    """Validates data against defined schemas."""


    def __init__(self):
        """Initialize data validator."""
        if not JSONSCHEMA_AVAILABLE:
            logger.warning("jsonschema not available, validation will be limited")


    def validate_training_config(self, config: Dict) -> bool:
        """Validate training configuration.

        Args:
            config: Training configuration

        Returns:
            True if validation succeeds
        """
        return validate_schema(config, TRAINING_CONFIG_SCHEMA)


    def validate_model_config(self, config: Dict) -> bool:
        """Validate model configuration.

        Args:
            config: Model configuration

        Returns:
            True if validation succeeds
        """
        return validate_schema(config, MODEL_CONFIG_SCHEMA)


    def validate_data_config(self, config: Dict) -> bool:
        """Validate data configuration.

        Args:
            config: Data configuration

        Returns:
            True if validation succeeds
        """
        return validate_schema(config, DATA_CONFIG_SCHEMA)


    def validate_custom_schema(self, data: Dict, schema: Dict) -> bool:
        """Validate data against custom schema.

        Args:
            data: Data to validate
            schema: Custom JSON schema

        Returns:
            True if validation succeeds
        """
        return validate_schema(data, schema)


    def get_validation_errors(self, data: Dict, schema: Dict) -> List[str]:
        """Get validation errors.

        Args:
            data: Data to validate
            schema: JSON schema

        Returns:
            List of validation error messages
        """
        if not JSONSCHEMA_AVAILABLE:
            logger.warning("jsonschema not available, cannot get validation errors")
            return []

        try:
            validator = Draft7Validator(schema)
            return [error.message for error in validator.iter_errors(data)]
        except Exception as e:
            logger.error(f"Error getting validation errors: {e}")
            return [str(e)]
