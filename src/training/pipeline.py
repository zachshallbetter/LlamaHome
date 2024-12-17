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

from .cache import CacheConfig, CacheManager
from .data import DataConfig, DataManager
from .monitoring import MonitorConfig, MonitorManager
from .optimization import OptimizationConfig, Optimizer
from .processing import ProcessingConfig, TensorProcessor
from .resources import ResourceConfig, ResourceManager

@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    # Component configs
    cache: CacheConfig = CacheConfig()
    data: DataConfig = DataConfig()
    monitor: MonitorConfig = MonitorConfig()
    optimization: OptimizationConfig = OptimizationConfig()
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
    """Training pipeline with resource management and monitoring."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self._setup_pipeline()
    
    def _setup_pipeline(self) -> None:
        """Set up pipeline components."""
        # Set up paths
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cache_manager = CacheManager(
            self.config.cache_dir or self.output_dir / "cache",
            self.config.cache
        )
        
        self.data_manager = DataManager(
            self.tokenizer,
            self.config.data
        )
        
        self.monitor_manager = MonitorManager(
            self.config.monitor,
            self.output_dir / "logs"
        )
        
        self.optimizer = Optimizer(
            self.model,
            self.config.optimization
        )
        
        self.processor = TensorProcessor(
            self.model,
            self.config.processing
        )
        
        self.resource_manager = ResourceManager(
            self.config.resource
        )
    
    async def train(
        self,
        train_data: Union[str, Path, DataLoader],
        eval_data: Optional[Union[str, Path, DataLoader]] = None
    ) -> None:
        """
        Train model with monitoring and resource management.
        
        Args:
            train_data: Training data
            eval_data: Evaluation data
        """
        # Prepare data
        train_loader, eval_loader = await self._prepare_data(
            train_data,
            eval_data
        )
        
        # Calculate steps
        num_training_steps = self._calculate_training_steps(train_loader)
        
        # Set up optimization
        self.optimizer.setup_scheduler(num_training_steps)
        
        try:
            # Start monitoring
            await self.monitor_manager.start()
            
            # Training loop
            await self._training_loop(
                train_loader,
                eval_loader,
                num_training_steps
            )
            
        finally:
            # Stop monitoring
            await self.monitor_manager.stop()
    
    async def _prepare_data(
        self,
        train_data: Union[str, Path, DataLoader],
        eval_data: Optional[Union[str, Path, DataLoader]] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare training and evaluation data."""
        if isinstance(train_data, (str, Path)):
            train_loader, eval_loader = await self.data_manager.load_data(
                train_data
            )
            if eval_data is None:
                return train_loader, eval_loader
        else:
            train_loader = train_data
        
        if isinstance(eval_data, (str, Path)):
            _, eval_loader = await self.data_manager.load_data(eval_data)
        else:
            eval_loader = eval_data
        
        return train_loader, eval_loader
    
    def _calculate_training_steps(self, train_loader: DataLoader) -> int:
        """Calculate total training steps."""
        if self.config.max_steps:
            return self.config.max_steps
            
        return len(train_loader) * self.config.num_epochs
    
    async def _training_loop(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        num_training_steps: int
    ) -> None:
        """Main training loop."""
        step = 0
        epoch = 0
        best_eval_loss = float("inf")
        no_improvement_count = 0
        
        while step < num_training_steps:
            # Training epoch
            for batch in train_loader:
                # Check resources
                await self.resource_manager.wait_for_resources()
                
                try:
                    # Process batch
                    metrics = await self.processor.process_batch(
                        batch,
                        is_training=True
                    )
                    
                    # Optimization step
                    self.optimizer.step(
                        metrics["loss"],
                        update_scheduler=True
                    )
                    
                    # Update metrics
                    metrics["learning_rate"] = self.optimizer.get_last_lr()[0]
                    self.monitor_manager.update(step, metrics)
                    
                    # Save checkpoint
                    if step > 0 and step % self.config.save_steps == 0:
                        await self._save_checkpoint(step, metrics)
                    
                    # Evaluation
                    if (
                        eval_loader is not None and
                        step > 0 and
                        step % self.config.eval_steps == 0
                    ):
                        eval_metrics = await self._evaluate(eval_loader)
                        self.monitor_manager.update(step, eval_metrics)
                        
                        # Early stopping
                        if eval_metrics["loss"] < best_eval_loss - self.config.early_stopping_threshold:
                            best_eval_loss = eval_metrics["loss"]
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                            
                        if no_improvement_count >= self.config.early_stopping_patience:
                            print("Early stopping triggered")
                            return
                    
                    step += 1
                    if step >= num_training_steps:
                        break
                        
                finally:
                    # Release resources
                    self.resource_manager.release_resources()
            
            epoch += 1
            if epoch >= self.config.num_epochs:
                break
    
    async def _evaluate(
        self,
        eval_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        total_metrics = {}
        num_batches = 0
        
        for batch in eval_loader:
            # Process batch
            metrics = await self.processor.process_batch(
                batch,
                is_training=False
            )
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value.item()
            
            num_batches += 1
        
        # Average metrics
        return {
            f"eval_{key}": value / num_batches
            for key, value in total_metrics.items()
        }
    
    async def _save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, torch.Tensor]
    ) -> None:
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer and scheduler
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )
        
        # Save metrics
        torch.save(
            {
                "metrics": {
                    key: value.item()
                    for key, value in metrics.items()
                },
                "step": step
            },
            checkpoint_dir / "metrics.pt"
        )

class TrainingError(Exception):
    """Training pipeline error."""
    pass 