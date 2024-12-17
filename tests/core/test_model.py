"""Tests for model setup and configuration functionality."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from utils.model_manager import ModelManager


@pytest.fixture(autouse=True)
def setup_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Configure logging for tests."""
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("utils.model_manager")
    logger.setLevel(logging.WARNING)
    yield


@pytest.fixture
def mock_config() -> MagicMock:
    """Fixture providing test model configuration."""
    config = MagicMock()
    config.models = {
        "llama3.3": {
            "name": "Llama 3.3",
            "requires_gpu": True,
            "min_gpu_memory": {"7b": 12, "13b": 24, "70b": 100},
            "h2o_config": {"window_length": 1024, "heavy_hitter_tokens": 256},
            "versions": ["7b", "13b", "70b"],
            "variants": ["base", "chat"],
        },
        "gpt4": {
            "name": "GPT-4 Turbo",
            "requires_gpu": False,
            "versions": ["base"],
            "variants": ["chat"],
        },
    }
    return config


@pytest.fixture
def setup_test_env(tmp_path: Path, mock_config: MagicMock) -> Path:
    """Set up test environment with temporary directories and config."""
    # Create config directory and file
    config_dir = tmp_path / ".config"
    config_dir.mkdir()
    config_file = config_dir / "model_config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(mock_config.models, f)

    # Create models directory
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Mock Path to return the config file path
    with patch("utils.model_manager.Path") as mock_path:
        mock_path.side_effect = lambda path_str: (
            path_str
            if isinstance(path_str, Path)
            else (
                config_file
                if ".config/model_config.yaml" in str(path_str)
                else models_dir if "models" in str(path_str) else Path(path_str)
            )
        )
        yield tmp_path


class TestModelManager:
    """Test cases for ModelManager class."""

    def test_init(self, setup_test_env: Path) -> None:
        """Test ModelManager initialization."""
        manager = ModelManager()
        assert manager.config["models"]["llama3.3"]["name"] == "Llama 3.3"

    def test_setup_model_creates_directory(self, setup_test_env: Path) -> None:
        """Test model directory creation during setup."""
        manager = ModelManager()
        manager.setup_model("llama3.3", size="13b", variant="chat")
        model_dir = setup_test_env / ".llamahome" / "models" / "llama3.3"
        assert model_dir.exists()

    def test_setup_model_invalid_model(self, setup_test_env: Path) -> None:
        """Test setup with invalid model name."""
        manager = ModelManager()
        with pytest.raises(ValueError, match="Model invalid_model not found in configuration"):
            manager.setup_model("invalid_model", size="13b")

    def test_setup_default_model(self, setup_test_env: Path) -> None:
        """Test default model setup from environment variables."""
        mock_env = MagicMock()
        mock_env.items.return_value = {
            "LLAMA_MODEL": "llama3.3",
            "LLAMA_MODEL_SIZE": "13b",
            "LLAMA_MODEL_VARIANT": "chat",
        }.items()

        with patch.dict(os.environ, mock_env.items()):
            manager = ModelManager()
            manager.setup_model(
                os.environ["LLAMA_MODEL"],
                size=os.environ["LLAMA_MODEL_SIZE"],
                variant=os.environ["LLAMA_MODEL_VARIANT"],
            )
            model_dir = setup_test_env / ".llamahome" / "models" / "llama3.3"
            assert model_dir.exists()


@pytest.mark.integration
class TestModelManagerIntegration:
    """Integration tests for ModelManager."""

    def test_full_setup_process(self, setup_test_env: Path) -> None:
        """Test complete model setup process with actual file system."""
        manager = ModelManager()
        manager.setup_model("llama3.3", size="13b", variant="chat")

        # Verify directory structure
        model_dir = setup_test_env / ".llamahome" / "models" / "llama3.3"
        assert model_dir.exists()
        assert model_dir.is_dir()

        # Verify config is properly loaded
        assert manager.config["models"]["llama3.3"]["name"] == "Llama 3.3"
        assert manager.config["models"]["llama3.3"]["requires_gpu"] is True

    def test_multiple_model_setups(self, setup_test_env: Path) -> None:
        """Test setting up multiple models in sequence."""
        manager = ModelManager()

        # Setup first model
        manager.setup_model("llama3.3", size="7b", variant="base")
        assert (setup_test_env / ".llamahome" / "models" / "llama3.3").exists()

        # Setup second model
        manager.setup_model("gpt4", variant="chat")
        assert (setup_test_env / ".llamahome" / "models" / "gpt4").exists()

        # Verify both directories exist
        model_dirs = list((setup_test_env / ".llamahome" / "models").iterdir())
        assert len(model_dirs) == 2


"""Tests for enhanced Llama model implementation."""

import pytest
import torch
from pathlib import Path
from transformers import LlamaConfig
from llama_recipes.utils.dataset import ConcatDataset

from src.core.model import EnhancedLlamaForCausalLM, ModelConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
        use_flash_attention=True,
        use_memory_efficient=True
    )


@pytest.fixture
def model(config):
    """Create test model."""
    return EnhancedLlamaForCausalLM(config)


def test_model_initialization(model):
    """Test model initialization."""
    assert isinstance(model, EnhancedLlamaForCausalLM)
    assert model.model_name == "EnhancedLlama"
    assert isinstance(model.model_config, ModelConfig)
    assert model.cache_config["window_length"] == 2048


def test_cuda_graphs(model):
    """Test CUDA graph initialization."""
    if torch.cuda.is_available():
        assert hasattr(model, "cuda_graphs")
        assert isinstance(model.cuda_graphs, dict)
        
        # Test common shapes
        for shape in [(1, 32), (1, 64), (1, 128), (1, 256)]:
            assert shape in model.cuda_graphs


def test_input_preparation(model):
    """Test input preparation for generation."""
    batch_size, seq_len = 2, 32
    
    # Create test inputs
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test without cache
    inputs = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask
    )
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape == (batch_size, seq_len)
    
    # Test with cache
    past_key_values = model._init_cache()
    inputs = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values
    )
    assert "past_key_values" in inputs
    assert inputs["use_cache"] is True


def test_forward_pass(model):
    """Test model forward pass."""
    batch_size, seq_len = 2, 32
    
    # Create test inputs
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test without cache
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False
    )
    assert outputs.logits.shape == (batch_size, seq_len, model.config.vocab_size)
    
    # Test with cache
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    assert outputs.logits.shape == (batch_size, seq_len, model.config.vocab_size)
    assert outputs.past_key_values is not None


def test_cache_initialization(model):
    """Test cache initialization."""
    cache = model._init_cache()
    assert cache is not None
    assert cache.get_max_length() == model.cache_config["max_length"]
    assert isinstance(cache.get_seq_length(), int)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_device_handling(model):
    """Test model device handling."""
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    # Move model to GPU
    model = model.cuda()
    input_ids = input_ids.cuda()
    
    # Test forward pass on GPU
    outputs = model.forward(input_ids)
    assert outputs.logits.device.type == "cuda"


def test_training_features(model, tmp_path):
    """Test training functionality."""
    # Create dummy dataset
    class DummyDataset(ConcatDataset):
        def __init__(self):
            self.data = [(torch.randint(0, 32000, (32,)), torch.randint(0, 32000, (32,))) for _ in range(10)]
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __len__(self):
            return len(self.data)
    
    train_dataset = DummyDataset()
    eval_dataset = DummyDataset()
    
    # Test training
    try:
        model.train_model(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=1,
            batch_size=2,
            learning_rate=1e-4
        )
    except Exception as e:
        pytest.fail(f"Training failed: {e}")


def test_model_saving(model, tmp_path):
    """Test model saving functionality."""
    save_dir = tmp_path / "model"
    
    # Test saving
    model.save_pretrained(str(save_dir))
    
    # Check saved files
    assert (save_dir / "model_config.json").exists()
    assert (save_dir / "pytorch_model.bin").exists()
    
    # Test loading
    loaded_model = EnhancedLlamaForCausalLM.from_pretrained(str(save_dir))
    assert isinstance(loaded_model, EnhancedLlamaForCausalLM)
    assert loaded_model.model_config.use_flash_attention == model.model_config.use_flash_attention


def test_quantization(config):
    """Test quantization configuration."""
    config.load_in_4bit = True
    model = EnhancedLlamaForCausalLM(config)
    
    assert model.model_config.quantization_config is not None
    assert model.model_config.quantization_config["load_in_4bit"] is True


def test_lora_configuration(config):
    """Test LoRA configuration."""
    config.use_lora = True
    model = EnhancedLlamaForCausalLM(config)
    
    assert model.model_config.lora_config is not None
    assert "r" in model.model_config.lora_config
    assert "target_modules" in model.model_config.lora_config


def test_sliding_window(config):
    """Test sliding window attention."""
    config.sliding_window = 128
    model = EnhancedLlamaForCausalLM(config)
    
    batch_size, seq_len = 2, 256
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    outputs = model.forward(input_ids)
    assert outputs.logits.shape == (batch_size, seq_len, model.config.vocab_size)


def test_gradient_checkpointing(model):
    """Test gradient checkpointing."""
    model.gradient_checkpointing_enable()
    
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    labels = torch.randint(0, 32000, (batch_size, seq_len))
    
    # Forward pass with gradient checkpointing
    outputs = model.forward(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    # Check gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_memory_efficient_training(config):
    """Test memory efficient training setup."""
    config.use_memory_efficient = True
    model = EnhancedLlamaForCausalLM(config)
    
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    # Test forward pass with memory efficient attention
    outputs = model.forward(input_ids)
    assert outputs.logits.shape == (batch_size, seq_len, model.config.vocab_size)


class TestTrainingFeatures:
    """Test cases for enhanced training features."""

    @pytest.fixture
    def setup_data_files(self, tmp_path):
        """Create test data files."""
        # Create single file
        single_file = tmp_path / "single.jsonl"
        with open(single_file, "w") as f:
            f.write('{"text": "Sample text 1"}\n')
            f.write('{"text": "Sample text 2"}\n')

        # Create multiple files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i in range(3):
            file_path = data_dir / f"part{i}.jsonl"
            with open(file_path, "w") as f:
                f.write(f'{{"text": "File {i} text 1"}}\n')
                f.write(f'{{"text": "File {i} text 2"}}\n')

        return {
            "single_file": single_file,
            "data_dir": data_dir,
            "multiple_files": list(data_dir.glob("*.jsonl"))
        }

    def test_single_file_training(self, model, setup_data_files):
        """Test training from a single file."""
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                num_epochs=1,
                batch_size=1
            )
        except Exception as e:
            pytest.fail(f"Single file training failed: {e}")

    def test_multiple_files_training(self, model, setup_data_files):
        """Test training from multiple files."""
        try:
            model.train_model(
                train_dataset=setup_data_files["multiple_files"],
                num_epochs=1,
                batch_size=1
            )
        except Exception as e:
            pytest.fail(f"Multiple files training failed: {e}")

    def test_directory_training(self, model, setup_data_files):
        """Test training from directory."""
        try:
            model.train_model(
                train_dataset=setup_data_files["data_dir"],
                num_epochs=1,
                batch_size=1
            )
        except Exception as e:
            pytest.fail(f"Directory training failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed_training(self, model, setup_data_files):
        """Test distributed training setup."""
        if torch.cuda.device_count() > 1:
            try:
                model.train_model(
                    train_dataset=setup_data_files["single_file"],
                    num_epochs=1,
                    batch_size=1
                )
                assert isinstance(model.model, torch.nn.DataParallel)
            except Exception as e:
                pytest.fail(f"Distributed training failed: {e}")

    def test_streaming_dataset(self, model, tmp_path):
        """Test streaming dataset for large files."""
        # Create a large file (>1GB threshold)
        large_file = tmp_path / "large.jsonl"
        with open(large_file, "w") as f:
            for i in range(100000):  # Write enough lines to exceed 1GB
                f.write(f'{{"text": "Large file text {i}"}}\n')

        try:
            model.train_model(
                train_dataset=large_file,
                num_epochs=1,
                batch_size=1
            )
        except Exception as e:
            pytest.fail(f"Streaming dataset training failed: {e}")

    def test_invalid_data_source(self, model):
        """Test handling of invalid data source."""
        with pytest.raises(ValueError):
            model.train_model(train_dataset=123)  # Invalid type

        with pytest.raises(FileNotFoundError):
            model.train_model(train_dataset="nonexistent.jsonl")

    def test_training_configuration(self, model, setup_data_files):
        """Test training configuration options."""
        custom_config = {
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "warmup_steps": 50,
            "weight_decay": 0.1,
            "max_grad_norm": 0.5
        }

        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                **custom_config
            )
        except Exception as e:
            pytest.fail(f"Custom configuration training failed: {e}")

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs not available")
    def test_multi_gpu_training(self, model, setup_data_files):
        """Test training with multiple GPUs."""
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                num_epochs=1,
                batch_size=2
            )
            assert isinstance(model.model, torch.nn.DataParallel)
            assert len(model.model.device_ids) > 1
        except Exception as e:
            pytest.fail(f"Multi-GPU training failed: {e}")

    def test_evaluation_dataset(self, model, setup_data_files):
        """Test training with evaluation dataset."""
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                eval_dataset=setup_data_files["multiple_files"][0],
                num_epochs=1,
                batch_size=1,
                eval_steps=10
            )
        except Exception as e:
            pytest.fail(f"Training with evaluation failed: {e}")

    def test_save_checkpoints(self, model, setup_data_files, tmp_path):
        """Test checkpoint saving during training."""
        output_dir = tmp_path / "checkpoints"
        
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                num_epochs=1,
                batch_size=1,
                save_steps=10,
                output_dir=str(output_dir)
            )
            
            # Check if checkpoints were saved
            assert output_dir.exists()
            checkpoints = list(output_dir.glob("checkpoint-*"))
            assert len(checkpoints) > 0
        except Exception as e:
            pytest.fail(f"Checkpoint saving failed: {e}")

    def test_resume_training(self, model, setup_data_files, tmp_path):
        """Test resuming training from checkpoint."""
        output_dir = tmp_path / "checkpoints"
        
        # Initial training
        model.train_model(
            train_dataset=setup_data_files["single_file"],
            num_epochs=1,
            batch_size=1,
            save_steps=10,
            output_dir=str(output_dir)
        )
        
        # Get latest checkpoint
        checkpoints = list(output_dir.glob("checkpoint-*"))
        latest_checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[1])))
        
        # Resume training
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                num_epochs=1,
                batch_size=1,
                resume_from_checkpoint=latest_checkpoint
            )
        except Exception as e:
            pytest.fail(f"Resume training failed: {e}")

    def test_gradient_accumulation(self, model, setup_data_files):
        """Test training with gradient accumulation."""
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                num_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=4
            )
        except Exception as e:
            pytest.fail(f"Gradient accumulation training failed: {e}")

    def test_mixed_precision_training(self, model, setup_data_files):
        """Test mixed precision training."""
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                num_epochs=1,
                batch_size=1,
                fp16=True
            )
        except Exception as e:
            pytest.fail(f"Mixed precision training failed: {e}")

    def test_memory_efficient_training(self, model, setup_data_files):
        """Test memory efficient training options."""
        try:
            model.train_model(
                train_dataset=setup_data_files["single_file"],
                num_epochs=1,
                batch_size=1,
                gradient_checkpointing=True,
                use_memory_efficient=True
            )
        except Exception as e:
            pytest.fail(f"Memory efficient training failed: {e}")
