"""Training manager utilities."""

from pathlib import Path
from typing import Dict, Optional, Union

from ..core.utils import LogManager, LogTemplates
from ..data.training import TrainingData

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class TrainingManager:
    """Manages training data processing and configuration."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 4,
        max_workers: int = 4,
        config: Optional[Dict] = None
    ) -> None:
        """Initialize training manager.

        Args:
            data_dir: Directory for training data
            batch_size: Size of training batches 
            max_workers: Maximum number of worker processes
            config: Optional configuration dictionary
        """
        self.training_data = TrainingData(data_dir, batch_size, max_workers, config)
        logger.info(f"Initialized training manager with data dir: {data_dir}")

    async def process_samples(
        self,
        model_name: str = "llama",
        model_version: Optional[str] = None
    ) -> None:
        """Process and prepare training samples.

        Args:
            model_name: Name of model to train
            model_version: Optional specific model version
        """
        await self.training_data.process_samples(model_name, model_version)

    async def train_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ) -> None:
        """Train model on processed data using LoRA fine-tuning.

        Args:
            model_name: Name of model to train
            model_version: Optional specific model version
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout value
        """
        await self.training_data.train_model(
            model_name,
            model_version,
            num_epochs,
            learning_rate,
            lora_r,
            lora_alpha,
            lora_dropout
        )


def create_training_manager(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    max_workers: int = 4,
    config: Optional[Dict] = None
) -> TrainingManager:
    """Create a new TrainingManager instance.

    Args:
        data_dir: Directory for training data
        batch_size: Size of training batches
        max_workers: Maximum number of worker processes
        config: Optional configuration dictionary

    Returns:
        Configured TrainingManager instance
    """
    return TrainingManager(data_dir, batch_size, max_workers, config)
