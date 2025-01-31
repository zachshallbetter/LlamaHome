"""Tests for training data management functionality."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from src.data.storage import DataStorage
from src.data.training import ConversationDataset
from src.training.manager import TrainingManager, create_training_manager


@pytest.fixture
async def storage(tmp_path: Path) -> DataStorage:
    """Create a temporary storage instance."""
    return DataStorage(base_path=tmp_path)


@pytest.fixture
async def training_manager(tmp_path: Path) -> TrainingManager:
    """Create a training manager instance for testing.

    Returns:
        TrainingManager instance
    """
    manager = create_training_manager(data_dir=tmp_path)
    return manager


@pytest.fixture
def sample_conversations() -> list[dict[str, Any]]:
    """Create sample conversation data."""
    return [
        {
            "conversation": [
                {"role": "user", "content": "What is LlamaHome?"},
                {"role": "assistant", "content": "LlamaHome is an AI assistant platform."}
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "How do I train a model?"},
                {"role": "assistant", "content": "You can use the training command."}
            ]
        }
    ]


@pytest.fixture
async def sample_file(tmp_path: Path, sample_conversations: list[dict[str, Any]]) -> Path:
    """Create a sample training file."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir(parents=True)
    file_path = samples_dir / "test_samples.jsonl"

    with open(file_path, "w") as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + "\n")

    return file_path


@pytest.mark.asyncio
async def test_process_samples(
    training_manager: TrainingManager,
    sample_file: Path,
    tmp_path: Path
) -> None:
    """Test processing training samples."""
    # Create mock model directory and files
    model_dir = tmp_path / ".cache" / "models" / "llama" / "test"
    model_dir.mkdir(parents=True)

    # Create mock tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for file in tokenizer_files:
        (model_dir / file).touch()

    # Process samples
    await training_manager.process_samples("llama", "test")

    # Verify processed data was saved
    processed_file = tmp_path / "processed" / "llama_test_data.pt"
    assert processed_file.exists()

    # Load and verify processed data
    data = torch.load(processed_file)
    assert "samples" in data
    assert "config" in data
    assert len(data["samples"]) == 2


@pytest.mark.asyncio
async def test_conversation_dataset(
    sample_conversations: list[dict[str, Any]],
    tmp_path: Path
) -> None:
    """Test conversation dataset functionality."""
    # Create mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT2 tokenizer for testing

    # Create dataset
    dataset = ConversationDataset(sample_conversations, tokenizer)

    # Test dataset size
    assert len(dataset) == 2

    # Test getting items
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].dim() == 1
    assert item["attention_mask"].dim() == 1


@pytest.mark.asyncio
async def test_training_configuration(
    training_manager: TrainingManager
) -> None:
    """Test training configuration parameters."""
    assert training_manager.training_data.batch_size == 4
    assert training_manager.training_data.max_workers == 4
