"""
Main training pipeline implementation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.utils import LogManager, LogTemplates
from ..core.utils.io import safe_torch_load, safe_torch_save
from .cache import CacheConfig, CacheManager
from .data import DataConfig, DataManager
from .monitoring import MetricsConfig, MonitorManager
from .optimization import Optimizer, OptimizerConfig, SchedulerConfig
from .processing import ProcessingConfig, ProcessingError, TensorProcessor
from .resources import ResourceConfig, ResourceManager

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    # Basic params
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3

    # Optimization
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    # Data configuration
    data: DataConfig = DataConfig()
    cache_config: CacheConfig = CacheConfig()
    monitor_config: MetricsConfig = MetricsConfig()
    processing_config: ProcessingConfig = ProcessingConfig()
    resource_config: ResourceConfig = ResourceConfig()

    # Training steps
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 1000
    max_steps: Optional[int] = None
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Paths and directories
    output_dir: str = "output"
    cache_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    tensorboard_dir: str = "runs"

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    # Mixed precision training
    fp16: bool = False
    fp16_opt_level: str = "O1"
    fp16_backend: str = "auto"
    fp16_full_eval: bool = False

    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    ddp_backend: str = "nccl"
    gradient_checkpointing: bool = False
    sync_bn: bool = False
    find_unused_parameters: bool = False

    # Model configuration
    model_revision: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: bool = False
    low_cpu_mem_usage: bool = False
    torch_dtype: Optional[str] = None

    # Memory optimization
    max_memory: Optional[Dict[int, str]] = None
    offload_folder: Optional[str] = None
    offload_state_dict: bool = False

    def __post_init__(self) -> None:
        """Validate and set derived configurations."""
        # Set checkpoint directory if not specified
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

        # Validate mixed precision settings
        if self.fp16 and self.torch_dtype is None:
            self.torch_dtype = "float16"

        # Validate distributed settings
        if self.world_size > 1 and self.local_rank == -1:
            raise ValueError("Must specify local_rank for distributed training")


class TrainingPipeline:
    """Main training pipeline implementation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[TrainingConfig] = None,
    ):
        """Initialize training pipeline.

        Args:
            model: Model to train
            tokenizer: Tokenizer instance
            config: Optional training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.optimizer = Optimizer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            config=self.config.optimizer,
        )
        self._setup_components()

    def _setup_components(self) -> None:
        """Set up training components."""
        # Cache manager
        self.cache_manager = CacheManager(
            cache_dir=Path(self.config.cache_dir) if self.config.cache_dir else None,
            config=self.config.cache_config,
        )

        # Data manager
        self.data_manager = DataManager(
            tokenizer=self.tokenizer, config=self.config.data_config
        )

        # Monitor manager
        self.monitor_manager = MonitorManager(
            config=self.config.monitor_config, model_name=self.model.__class__.__name__
        )

        # Processor
        self.processor = TensorProcessor(
            model=self.model, config=self.config.processing_config
        )

        # Resource manager
        self.resource_manager = ResourceManager(config=self.config.resource_config)

    async def train(
        self, train_data: Union[str, Path], eval_data: Optional[Union[str, Path]] = None
    ) -> None:
        """Run training pipeline.

        Args:
            train_data: Training data path
            eval_data: Optional evaluation data path
        """
        try:
            # Prepare data
            train_dataset = await self.data_manager.prepare_dataset(train_data)
            eval_dataset = (
                await self.data_manager.prepare_dataset(eval_data)
                if eval_data
                else None
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.data.batch_size, shuffle=True
            )

            eval_loader = (
                DataLoader(eval_dataset, batch_size=self.config.data.batch_size)
                if eval_dataset
                else None
            )

            # Setup training
            self.optimizer.setup_scheduler(
                num_training_steps=len(train_loader) * self.config.num_epochs
            )

            # Training loop
            for epoch in range(self.config.num_epochs):
                await self._train_epoch(epoch, train_loader, eval_loader)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            await self.cache_manager.clear()

    async def _train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ) -> None:
        """Run single training epoch.

        Args:
            epoch: Current epoch
            train_loader: Training data loader
            eval_loader: Optional evaluation data loader
        """
        self.model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            # Process batch
            inputs = self.processor.prepare_inputs(batch)

            # Forward pass
            outputs = self.model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            # Optimization step
            self.optimizer.step(loss)

            # Logging
            if step % self.config.logging_steps == 0:
                await self.monitor_manager.update_metrics(
                    step + epoch * len(train_loader),
                    {
                        "loss": loss.item(),
                        "learning_rate": self.optimizer.get_last_lr()[0],
                    },
                    self.model,
                )

            # Evaluation
            if eval_loader and step % self.config.eval_steps == 0:
                eval_loss = await self._evaluate(eval_loader)
                await self.monitor_manager.update_metrics(
                    step + epoch * len(train_loader), {"eval_loss": eval_loss}
                )

            # Save checkpoint
            if step % self.config.save_steps == 0:
                await self._save_checkpoint(epoch, step, total_loss / (step + 1))

    async def _evaluate(self, eval_loader: DataLoader) -> float:
        """Run evaluation.

        Args:
            eval_loader: Evaluation data loader

        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_loader:
                inputs = self.processor.prepare_inputs(batch)
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()

        return total_loss / len(eval_loader)

    async def _save_checkpoint(self, epoch: int, step: int, loss: float) -> None:
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint-{epoch}-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        safe_torch_save(state_dict, checkpoint_path / "training_state.pt")

    async def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        # Load model and tokenizer
        self.model = self.model.from_pretrained(checkpoint_path)
        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_path)

        # Load training state
        state_dict = safe_torch_load(checkpoint_path / "training_state.pt")
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.config = state_dict["config"]

        logger.info(f"Loaded checkpoint: {checkpoint_path}")


class TrainingError(Exception):
    """Training pipeline error."""

    pass
