"""
Tests for training pipeline components.
"""

import asyncio
from pathlib import Path
from typing import Dict, List

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training import (
    CacheConfig,
    DataConfig,
    MonitorConfig,
    OptimizationConfig,
    ProcessingConfig,
    ResourceConfig,
    TrainingConfig,
    TrainingPipeline
)

@pytest.fixture
def model():
    """Get test model."""
    return AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        torch_dtype=torch.float16,
        device_map="auto"
    )

@pytest.fixture
def tokenizer():
    """Get test tokenizer."""
    return AutoTokenizer.from_pretrained("facebook/opt-125m")

@pytest.fixture
def config():
    """Get test configuration."""
    return TrainingConfig(
        cache=CacheConfig(
            memory_size=100,
            disk_size=1000
        ),
        data=DataConfig(
            batch_size=2,
            max_length=128,
            num_workers=0
        ),
        monitor=MonitorConfig(
            tensorboard=False,
            progress_bars=False
        ),
        optimization=OptimizationConfig(
            learning_rate=1e-5,
            warmup_steps=10
        ),
        processing=ProcessingConfig(
            mixed_precision=True,
            max_batch_size=2
        ),
        resource=ResourceConfig(
            gpu_memory_fraction=0.5
        ),
        num_epochs=1,
        save_steps=10,
        eval_steps=5,
        max_steps=20
    )

@pytest.fixture
def train_data():
    """Get test training data."""
    return [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello!"
                },
                {
                    "role": "assistant",
                    "content": "Hi there! How can I help you today?"
                }
            ]
        }
    ] * 10

@pytest.fixture
def eval_data():
    """Get test evaluation data."""
    return [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "How are you?"
                },
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking!"
                }
            ]
        }
    ] * 5

@pytest.mark.asyncio
async def test_training_pipeline_initialization(
    model,
    tokenizer,
    config,
    tmp_path
):
    """Test training pipeline initialization."""
    config.output_dir = str(tmp_path)
    pipeline = TrainingPipeline(model, tokenizer, config)
    
    assert pipeline.model == model
    assert pipeline.tokenizer == tokenizer
    assert pipeline.config == config
    assert Path(pipeline.output_dir).exists()

@pytest.mark.asyncio
async def test_training_with_dict_data(
    model,
    tokenizer,
    config,
    train_data,
    eval_data,
    tmp_path
):
    """Test training with dictionary data."""
    # Save test data
    train_path = tmp_path / "train.json"
    eval_path = tmp_path / "eval.json"
    
    import json
    with open(train_path, "w") as f:
        json.dump(train_data, f)
    with open(eval_path, "w") as f:
        json.dump(eval_data, f)
    
    # Initialize pipeline
    config.output_dir = str(tmp_path)
    pipeline = TrainingPipeline(model, tokenizer, config)
    
    # Train
    await pipeline.train(train_path, eval_path)
    
    # Check outputs
    assert (tmp_path / "checkpoint-10").exists()
    assert (tmp_path / "checkpoint-10" / "pytorch_model.bin").exists()
    assert (tmp_path / "checkpoint-10" / "optimizer.pt").exists()
    assert (tmp_path / "checkpoint-10" / "metrics.pt").exists()

@pytest.mark.asyncio
async def test_training_with_dataloader(
    model,
    tokenizer,
    config,
    train_data,
    eval_data,
    tmp_path
):
    """Test training with DataLoader."""
    from src.training import ConversationDataset
    
    # Create datasets
    train_dataset = ConversationDataset(
        train_data,
        tokenizer,
        max_length=config.data.max_length
    )
    eval_dataset = ConversationDataset(
        eval_data,
        tokenizer,
        max_length=config.data.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.data.batch_size,
        shuffle=False
    )
    
    # Initialize pipeline
    config.output_dir = str(tmp_path)
    pipeline = TrainingPipeline(model, tokenizer, config)
    
    # Train
    await pipeline.train(train_loader, eval_loader)
    
    # Check outputs
    assert (tmp_path / "checkpoint-10").exists()
    assert (tmp_path / "checkpoint-10" / "pytorch_model.bin").exists()
    assert (tmp_path / "checkpoint-10" / "optimizer.pt").exists()
    assert (tmp_path / "checkpoint-10" / "metrics.pt").exists()

@pytest.mark.asyncio
async def test_early_stopping(
    model,
    tokenizer,
    config,
    train_data,
    eval_data,
    tmp_path
):
    """Test early stopping."""
    # Configure early stopping
    config.output_dir = str(tmp_path)
    config.early_stopping_patience = 2
    config.early_stopping_threshold = 0.1
    config.eval_steps = 2
    
    # Initialize pipeline
    pipeline = TrainingPipeline(model, tokenizer, config)
    
    # Train
    await pipeline.train(train_data, eval_data)
    
    # Check if training stopped early
    checkpoints = list(tmp_path.glob("checkpoint-*"))
    assert len(checkpoints) < config.max_steps // config.save_steps

@pytest.mark.asyncio
async def test_resource_management(
    model,
    tokenizer,
    config,
    train_data,
    tmp_path
):
    """Test resource management during training."""
    # Configure resource limits
    config.output_dir = str(tmp_path)
    config.resource.gpu_memory_fraction = 0.1
    config.resource.cpu_usage_threshold = 0.5
    
    # Initialize pipeline
    pipeline = TrainingPipeline(model, tokenizer, config)
    
    # Train
    await pipeline.train(train_data)
    
    # Check resource usage
    gpu_info = pipeline.resource_manager.resources["gpu"].get_memory_info()
    assert gpu_info["allocated"] / gpu_info["total"] <= 0.1

@pytest.mark.asyncio
async def test_error_handling(
    model,
    tokenizer,
    config,
    tmp_path
):
    """Test error handling during training."""
    # Invalid data
    invalid_data = [{"invalid": "data"}]
    
    # Initialize pipeline
    config.output_dir = str(tmp_path)
    pipeline = TrainingPipeline(model, tokenizer, config)
    
    # Check error handling
    with pytest.raises(Exception):
        await pipeline.train(invalid_data) 