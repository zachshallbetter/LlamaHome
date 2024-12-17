"""
Data loading implementation for training pipeline.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

@dataclass
class DataConfig:
    """Data loading configuration."""
    batch_size: int = 32
    max_length: int = 512
    num_workers: int = 4
    shuffle: bool = True
    validation_split: float = 0.1
    cache_dir: Optional[str] = None
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    stream_buffer_size: int = 1000
    memory_limit: Optional[int] = None  # In MB

class StreamingDataset(Dataset):
    """Memory-efficient streaming dataset."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        buffer_size: int = 1000,
        memory_limit: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.memory_limit = memory_limit
        self._buffer = []
        self._file_iter = None
        self._total_size = self._count_samples()
        self._setup_streaming()
    
    def _count_samples(self) -> int:
        """Count total samples without loading into memory."""
        count = 0
        with self.data_path.open() as f:
            for _ in f:
                count += 1
        return count
    
    def _setup_streaming(self) -> None:
        """Set up file streaming."""
        self._file = self.data_path.open()
        self._file_iter = iter(self._file)
        self._refill_buffer()
    
    def _refill_buffer(self) -> None:
        """Refill buffer with next batch of samples."""
        while len(self._buffer) < self.buffer_size:
            try:
                line = next(self._file_iter)
                sample = json.loads(line)
                self._buffer.append(sample)
            except StopIteration:
                self._file.seek(0)
                self._file_iter = iter(self._file)
                if not self._buffer:
                    raise RuntimeError("Empty dataset")
                break
            
            if self.memory_limit:
                current_memory = sum(sys.getsizeof(s) for s in self._buffer)
                if current_memory > self.memory_limit * 1024 * 1024:  # Convert MB to bytes
                    break
    
    def __len__(self) -> int:
        return self._total_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized conversation with memory-efficient streaming."""
        if idx >= len(self._buffer):
            self._refill_buffer()
        
        conversation = self._buffer[idx % len(self._buffer)]
        
        # Format and tokenize efficiently
        text = self._format_conversation(conversation)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare labels efficiently
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def _format_conversation(self, conversation: Dict) -> str:
        """Format conversation for model input."""
        formatted = []
        
        for message in conversation["messages"]:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)

class DataManager:
    """Data management for training pipeline."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Optional[DataConfig] = None
    ):
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        self._setup_cache()
        self._setup_memory_tracking()
    
    def _setup_memory_tracking(self) -> None:
        """Set up memory usage tracking."""
        self.memory_tracker = MemoryTracker()
    
    async def load_data(
        self,
        data_path: Union[str, Path]
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Load and prepare training data with memory optimization."""
        # Create streaming datasets
        train_dataset = StreamingDataset(
            data_path,
            self.tokenizer,
            self.config.max_length,
            self.config.stream_buffer_size,
            self.config.memory_limit
        )
        
        # Split validation if needed
        val_dataset = None
        if self.config.validation_split > 0:
            val_size = int(len(train_dataset) * self.config.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create optimized data loaders
        train_loader = self._create_optimized_loader(
            train_dataset,
            shuffle=self.config.shuffle
        )
        
        val_loader = None
        if val_dataset:
            val_loader = self._create_optimized_loader(
                val_dataset,
                shuffle=False
            )
        
        return train_loader, val_loader
    
    def _create_optimized_loader(
        self,
        dataset: Dataset,
        shuffle: bool = True
    ) -> DataLoader:
        """Create memory-optimized data loader."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=shuffle,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            generator=torch.Generator(),
            worker_init_fn=self._worker_init_fn
        )
    
    def _worker_init_fn(self, worker_id: int) -> None:
        """Initialize worker with memory limits."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
            if hasattr(dataset, 'memory_limit'):
                dataset.memory_limit = self.config.memory_limit // worker_info.num_workers if self.config.memory_limit else None
    
    def _setup_cache(self) -> None:
        """Set up data caching."""
        if self.config.cache_dir:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def _load_raw_data(
        self,
        data_path: Union[str, Path]
    ) -> List[Dict]:
        """Load raw data from file."""
        data_path = Path(data_path)
        
        # Check cache first
        if self.config.cache_dir:
            cache_path = self.cache_dir / f"{data_path.stem}.cache"
            if cache_path.exists():
                return torch.load(cache_path)
        
        # Load and parse data
        try:
            with open(data_path, "r") as f:
                if data_path.suffix == ".json":
                    data = json.load(f)
                elif data_path.suffix == ".jsonl":
                    data = [json.loads(line) for line in f]
                else:
                    raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
            # Validate data
            self._validate_data(data)
            
            # Cache if enabled
            if self.config.cache_dir:
                torch.save(data, cache_path)
            
            return data
            
        except Exception as e:
            raise DataError(f"Failed to load data: {e}") from e
    
    def _validate_data(self, data: List[Dict]) -> None:
        """Validate data format."""
        for item in data:
            if "messages" not in item:
                raise DataError("Missing 'messages' field in conversation")
                
            for message in item["messages"]:
                if "role" not in message:
                    raise DataError("Missing 'role' field in message")
                if "content" not in message:
                    raise DataError("Missing 'content' field in message")
                if message["role"] not in ["system", "user", "assistant"]:
                    raise DataError(f"Invalid role: {message['role']}")
    
    def _split_data(
        self,
        data: List[Dict]
    ) -> Tuple[List[Dict], Optional[List[Dict]]]:
        """Split data into train and validation sets."""
        if not self.config.validation_split:
            return data, None
            
        split_idx = int(len(data) * (1 - self.config.validation_split))
        return data[:split_idx], data[split_idx:]
    
    def _create_loader(
        self,
        dataset: Dataset,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader with configuration."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=shuffle,
            pin_memory=True
        )

class DataError(Exception):
    """Data loading error."""
    pass 