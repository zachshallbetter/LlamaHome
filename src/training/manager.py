"""Training manager utilities."""

from pathlib import Path
from typing import Dict, Optional, Union

from ..core.config import DistributedConfig, TrainingConfig
from ..core.utils import LogManager, LogTemplates
from .pipeline import TrainingPipeline

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class TrainingManager:
    """Manages training data processing and configuration."""

    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
    ) -> None:
        """Initialize training manager."""
        self.training_config = training_config or TrainingConfig()
        self.distributed_config = distributed_config or DistributedConfig()
        self._validate_configs()

    def _validate_configs(self) -> None:
        """Validate configuration compatibility."""
        if self.distributed_config.basic["world_size"] > 1:
            if not self.training_config.optimization["gradient_checkpointing"]:
                raise ValueError("Distributed training requires gradient checkpointing")

    async def setup_training(self, model_path: Path) -> TrainingPipeline:
        """Set up training pipeline."""
        pipeline = TrainingPipeline(
            config=self.training_config, distributed_config=self.distributed_config
        )
        await pipeline.load_model(model_path)
        return pipeline

    async def process_samples(
        self, model_name: str = "llama", model_version: Optional[str] = None
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
        lora_dropout: float = 0.1,
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
            lora_dropout,
        )


def create_training_manager(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    max_workers: int = 4,
    config: Optional[Dict] = None,
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
