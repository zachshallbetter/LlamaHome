"""Basic inference example."""

from src.core.config import ConfigManager
from src.inference import InferenceConfig, InferenceManager


def main() -> None:
    """Run basic inference example."""
    # Get configuration
    ConfigManager()
    inference_config = InferenceConfig()

    # Initialize inference manager
    manager = InferenceManager(inference_config)

    # Run inference
    text = "Summarize this article:"
    result = manager.generate(text)
    print(result)


if __name__ == "__main__":
    main()
