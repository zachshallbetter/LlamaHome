"""
Main training pipeline implementation.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.utils import LogManager, LogTemplates
from .monitoring import TrainingMetrics
from .optimization import OptimizerConfig
from .data import DataConfig
from .cache import CacheConfig
from .monitoring import MetricsConfig
from .processing import ProcessingConfig
from .resource import ResourceConfig

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)

@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    # Component configs
    cache: CacheConfig = CacheConfig()
    data: DataConfig = DataConfig()
    monitor: MetricsConfig = MetricsConfig()
    optimization: OptimizerConfig = OptimizerConfig()
    processing: ProcessingConfig = ProcessingConfig()
    resource: ResourceConfig = ResourceConfig()
    
    # Training params
    num_epochs: int = 3
    save_steps: int = 1000
    eval_steps: int = 100
    logging_steps: int = 10
    max_steps: Optional[int] = None
    
    # Paths
    output_dir: str = "output"
    cache_dir: Optional[str] = None
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

class TrainingPipeline:
    """Main training pipeline implementation."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[TrainingConfig] = None
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
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self) -> None:
        """Set up training components."""
        # Cache manager
        self.cache_manager = CacheManager(
            cache_dir=self.config.cache_dir,
            config=self.config.cache
        )
        
        # Data manager
        self.data_manager = DataManager(
            tokenizer=self.tokenizer,
            config=self.config.data
        )
        
        # Monitor manager
        self.monitor_manager = MonitorManager(
            config=self.config.monitor,
            model_name=self.model.__class__.__name__
        )
        
        # Optimizer
        self.optimizer = Optimizer(
            config=self.config.optimization
        )
        
        # Processor
        self.processor = TensorProcessor(
            config=self.config.processing
        )
        
        # Resource manager
        self.resource_manager = ResourceManager(
            config=self.config.resource
        )
    
    async def train(
        self,
        train_data: Union[str, Path],
        eval_data: Optional[Union[str, Path]] = None
    ) -> None:
        """Run training pipeline.
        
        Args:
            train_data: Training data path
            eval_data: Optional evaluation data path
        """
        try:
            # Prepare data
            train_dataset = await self.data_manager.prepare_dataset(train_data)
            eval_dataset = await self.data_manager.prepare_dataset(eval_data) if eval_data else None
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True
            )
            
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.data.batch_size
            ) if eval_dataset else None
            
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
        eval_loader: Optional[DataLoader] = None
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
                        "learning_rate": self.optimizer.get_last_lr()[0]
                    },
                    self.model
                )
            
            # Evaluation
            if eval_loader and step % self.config.eval_steps == 0:
                eval_loss = await self._evaluate(eval_loader)
                await self.monitor_manager.update_metrics(
                    step + epoch * len(train_loader),
                    {"eval_loss": eval_loss}
                )
            
            # Save checkpoint
            if step % self.config.save_steps == 0:
                await self._save_checkpoint(
                    epoch,
                    step,
                    total_loss / (step + 1)
                )
    
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
    
    async def _save_checkpoint(
        self,
        epoch: int,
        step: int,
        loss: float
    ) -> None:
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Current loss
        """
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint-{epoch}-{step}"
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "optimizer": self.optimizer.state_dict(),
                "config": self.config
            },
            checkpoint_path / "training_state.pt"
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
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
        training_state = torch.load(checkpoint_path / "training_state.pt")
        self.optimizer.load_state_dict(training_state["optimizer"])
        self.config = training_state["config"]
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

class TrainingError(Exception):
    """Training pipeline error."""
    pass