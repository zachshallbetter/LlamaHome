"""JSON schema definitions and utilities."""

from typing import Dict

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from ...core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

# Training configuration schema
TRAINING_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "path": {"type": "string"}
            },
            "required": ["name", "type"]
        },
        "training": {
            "type": "object",
            "properties": {
                "batch_size": {"type": "integer", "minimum": 1},
                "epochs": {"type": "integer", "minimum": 1},
                "learning_rate": {"type": "number", "minimum": 0},
                "optimizer": {"type": "string"},
                "scheduler": {"type": "string"}
            },
            "required": ["batch_size", "epochs", "learning_rate"]
        },
        "data": {
            "type": "object",
            "properties": {
                "train_path": {"type": "string"},
                "val_path": {"type": "string"},
                "test_path": {"type": "string"}
            },
            "required": ["train_path"]
        }
    },
    "required": ["model", "training", "data"]
}

# Model configuration schema
MODEL_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "architecture": {"type": "string"},
        "hidden_size": {"type": "integer", "minimum": 1},
        "num_layers": {"type": "integer", "minimum": 1},
        "num_heads": {"type": "integer", "minimum": 1},
        "dropout": {"type": "number", "minimum": 0, "maximum": 1},
        "activation": {"type": "string"}
    },
    "required": ["architecture", "hidden_size", "num_layers", "num_heads"]
}

# Data configuration schema
DATA_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "format": {"type": "string"},
        "max_length": {"type": "integer", "minimum": 1},
        "preprocessing": {
            "type": "array",
            "items": {"type": "string"}
        },
        "augmentation": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["format", "max_length"]
}

def validate_schema(data: Dict, schema: Dict) -> bool:
    """Validate data against schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
        
    Returns:
        True if validation succeeds
    """
    if not JSONSCHEMA_AVAILABLE:
        logger.warning("jsonschema not available, skipping validation")
        return True
        
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Schema validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during schema validation: {e}")
        return False
