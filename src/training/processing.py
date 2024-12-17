"""
Tensor processing implementation for training pipeline.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

@dataclass
class ProcessingConfig:
    """Processing configuration."""
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    compile_mode: str = "reduce-overhead"
    max_batch_size: int = 32
    accumulation_steps: int = 4
    memory_efficient_attention: bool = True
    optimize_cuda_cache: bool = True
    gradient_clipping: float = 1.0
    cpu_offload: bool = False
    dynamic_batch_size: bool = True

class TensorProcessor:
    """Memory-efficient tensor processing."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ProcessingConfig] = None
    ):
        self.model = model
        self.config = config or ProcessingConfig()
        self._setup_processing()
        self._setup_memory_optimization()
    
    def _setup_memory_optimization(self) -> None:
        """Set up memory optimizations."""
        if self.config.optimize_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
        
        if self.config.memory_efficient_attention:
            self._enable_memory_efficient_attention()
        
        if self.config.cpu_offload:
            self._setup_cpu_offload()
    
    def _enable_memory_efficient_attention(self) -> None:
        """Enable memory efficient attention mechanism."""
        if hasattr(self.model, "config"):
            self.model.config.use_memory_efficient_attention = True
            self.model.config.use_scaled_dot_product_attention = True
    
    def _setup_cpu_offload(self) -> None:
        """Set up CPU offloading for memory optimization."""
        if hasattr(self.model, "to"):
            self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.offload_device = torch.device("cpu")
            self.model.to(self.model_device)
    
    async def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Process batch with memory optimization."""
        if self.config.dynamic_batch_size:
            batch = await self._adjust_batch_size(batch)
        
        if is_training:
            return await self._process_training_batch(batch)
        return await self._process_inference_batch(batch)
    
    async def _adjust_batch_size(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dynamically adjust batch size based on memory."""
        if not torch.cuda.is_available():
            return batch
            
        try:
            current_memory = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_ratio = current_memory / total_memory
            
            if memory_ratio > 0.8:  # Memory usage too high
                current_batch_size = next(iter(batch.values())).size(0)
                new_batch_size = max(1, current_batch_size // 2)
                
                return {
                    key: tensor[:new_batch_size]
                    for key, tensor in batch.items()
                }
        except Exception:
            pass  # Fall back to original batch on error
            
        return batch
    
    async def _process_training_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process training batch with memory optimization."""
        accumulated_loss = 0
        accumulated_metrics = {}
        
        # Split batch for gradient accumulation
        sub_batches = self._split_batch(batch)
        
        for i, sub_batch in enumerate(sub_batches):
            # Process with appropriate precision
            if self.config.mixed_precision:
                sub_results = await self._process_mixed_precision(
                    sub_batch,
                    is_training=True
                )
            else:
                sub_results = await self._process_full_precision(
                    sub_batch,
                    is_training=True
                )
            
            # Scale loss for gradient accumulation
            sub_results["loss"] = sub_results["loss"] / len(sub_batches)
            
            # Accumulate results
            accumulated_loss += sub_results["loss"].item()
            for key, value in sub_results.items():
                if key != "loss":
                    accumulated_metrics[key] = (
                        accumulated_metrics.get(key, 0) + value.item()
                    )
            
            # Clear memory after each sub-batch
            if self.config.optimize_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average results
        results = {
            "loss": accumulated_loss,
            **{
                key: value / len(sub_batches)
                for key, value in accumulated_metrics.items()
            }
        }
        
        return results
    
    async def _process_inference_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process inference batch."""
        with torch.no_grad():
            if self.config.mixed_precision:
                return await self._process_mixed_precision(
                    batch,
                    is_training=False
                )
            return await self._process_full_precision(
                batch,
                is_training=False
            )
    
    async def _process_mixed_precision(
        self,
        batch: Dict[str, torch.Tensor],
        is_training: bool
    ) -> Dict[str, torch.Tensor]:
        """Process batch with mixed precision and memory optimization."""
        with autocast():
            outputs = await self._forward_pass(batch, is_training)
            
            if is_training and self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clipping
                )
            
            return outputs
    
    async def _process_full_precision(
        self,
        batch: Dict[str, torch.Tensor],
        is_training: bool
    ) -> Dict[str, torch.Tensor]:
        """Process batch with full precision."""
        return await self._forward_pass(batch, is_training)
    
    async def _forward_pass(
        self,
        batch: Dict[str, torch.Tensor],
        is_training: bool
    ) -> Dict[str, torch.Tensor]:
        """Execute memory-efficient forward pass."""
        try:
            # Move batch to appropriate device
            if self.config.cpu_offload:
                batch = self._move_to_device(batch, self.model_device)
            else:
                batch = self._move_to_device(batch, next(self.model.parameters()).device)
            
            # Forward pass with memory tracking
            outputs = self.model(**batch)
            
            # Calculate metrics
            metrics = self._calculate_metrics(outputs, batch)
            
            # Optimize memory usage
            if self.config.cpu_offload and is_training:
                self._optimize_memory_usage(outputs, metrics)
            
            return metrics
            
        except Exception as e:
            raise ProcessingError(f"Forward pass failed: {e}") from e
    
    def _move_to_device(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Move batch to device with memory optimization."""
        return {
            key: tensor.to(device, non_blocking=True)
            for key, tensor in batch.items()
        }
    
    def _optimize_memory_usage(
        self,
        outputs: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor]
    ) -> None:
        """Optimize memory usage during training."""
        # Move unnecessary tensors to CPU
        for key, tensor in outputs.items():
            if key not in ["loss"]:
                outputs[key] = tensor.to(self.offload_device)
        
        for key, tensor in metrics.items():
            if key not in ["loss"]:
                metrics[key] = tensor.to(self.offload_device)
    
    def _split_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Split batch into sub-batches if needed."""
        batch_size = next(iter(batch.values())).size(0)
        
        if batch_size <= self.config.max_batch_size:
            return [batch]
        
        num_splits = (batch_size - 1) // self.config.max_batch_size + 1
        sub_batches = []
        
        for i in range(num_splits):
            start_idx = i * self.config.max_batch_size
            end_idx = min((i + 1) * self.config.max_batch_size, batch_size)
            
            sub_batch = {
                key: tensor[start_idx:end_idx]
                for key, tensor in batch.items()
            }
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    def _calculate_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate training metrics."""
        metrics = {}
        
        # Loss
        if hasattr(outputs, "loss"):
            metrics["loss"] = outputs.loss
        
        # Perplexity
        if "logits" in outputs and "labels" in batch:
            metrics["perplexity"] = self._calculate_perplexity(
                outputs.logits,
                batch["labels"]
            )
        
        # Accuracy
        if "logits" in outputs and "labels" in batch:
            metrics["accuracy"] = self._calculate_accuracy(
                outputs.logits,
                batch["labels"]
            )
        
        return metrics
    
    def _calculate_perplexity(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate perplexity metric."""
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        return torch.exp(loss)
    
    def _calculate_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate accuracy metric."""
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).float()
        return correct.mean()

class ProcessingError(Exception):
    """Processing error."""
    pass 