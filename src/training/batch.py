"""Batch generation and processing functionality."""

import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class BatchConfig:
    """Configuration for batch generation."""
    
    batch_size: int = 32
    max_sequence_length: int = 512
    dynamic_batching: bool = True
    drop_last: bool = False
    sort_by_length: bool = True
    pad_to_multiple: int = 8


class BatchGenerator:
    """Handles batch generation and processing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize batch generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = BatchConfig(**config.get("batch", {}))

    def generate_batch(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Generate a batch from samples.
        
        Args:
            samples: List of samples
            
        Returns:
            Batched data
        """
        if not samples:
            return {}

        # Sort by sequence length if enabled
        if self.config.sort_by_length and "input_ids" in samples[0]:
            samples = sorted(samples, key=lambda x: x["input_ids"].size(0), reverse=True)

        # Prepare batch
        batch = {}
        for key in samples[0].keys():
            if isinstance(samples[0][key], torch.Tensor):
                # Handle tensors
                if samples[0][key].dim() == 0:
                    # Scalar tensors
                    batch[key] = torch.stack([s[key] for s in samples])
                else:
                    # Pad sequence tensors
                    batch[key] = self._pad_sequence([s[key] for s in samples])
            else:
                # Handle non-tensor data
                batch[key] = [s[key] for s in samples]

        return batch

    def _pad_sequence(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length.
        
        Args:
            sequences: List of tensors to pad
            
        Returns:
            Padded tensor
        """
        max_len = max(seq.size(0) for seq in sequences)
        if self.config.pad_to_multiple > 1:
            max_len = ((max_len + self.config.pad_to_multiple - 1) 
                      // self.config.pad_to_multiple 
                      * self.config.pad_to_multiple)

        padded = []
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = [(0, max_len - seq.size(0))] + [(0, 0)] * (seq.dim() - 1)
                padded.append(torch.nn.functional.pad(seq, sum(padding[::-1], ())))
            else:
                padded.append(seq)

        return torch.stack(padded)

    def generate_dynamic_batch(
        self,
        samples: List[Dict[str, torch.Tensor]],
        max_tokens: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate dynamic batch based on sequence lengths.
        
        Args:
            samples: List of samples
            max_tokens: Optional maximum tokens per batch
            
        Returns:
            Dynamically batched data
        """
        if not self.config.dynamic_batching:
            return self.generate_batch(samples[:self.config.batch_size])

        if not max_tokens:
            max_tokens = self.config.batch_size * self.config.max_sequence_length

        # Sort by length for efficient batching
        samples = sorted(samples, key=lambda x: x["input_ids"].size(0), reverse=True)

        # Find largest batch that fits in max_tokens
        max_len = samples[0]["input_ids"].size(0)
        batch_size = min(
            max_tokens // max_len,
            self.config.batch_size,
            len(samples)
        )

        return self.generate_batch(samples[:batch_size])

    def collate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate samples into batch.
        
        Args:
            samples: List of samples
            
        Returns:
            Collated batch
        """
        if self.config.dynamic_batching:
            return self.generate_dynamic_batch(samples)
        return self.generate_batch(samples)
