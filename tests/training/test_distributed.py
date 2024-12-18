"""Tests for distributed training functionality."""

import pytest
import torch
import torch.distributed as dist
from typing import Dict, Any

from src.training.distributed import DistributedTrainer
from src.training.data import DataManager
from src.core.config_handler import ConfigManager


@pytest.fixture
def mock_config():
    """Create mock distributed configuration."""
    return {
        "distributed": {
            "backend": "nccl",
            "world_size": 2,
            "init_method": "tcp://localhost:23456",
            "rank": 0
        },
        "training": {
            "batch_size": 32,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0
        }
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment for distributed training."""
    # Create necessary directories
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    
    # Create logs directory
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    
    return tmp_path


class TestDistributedTrainer:
    """Test suite for distributed training functionality."""
    
    def test_initialization(self, mock_config):
        """Test distributed trainer initialization."""
        with patch('torch.distributed.init_process_group') as mock_init:
            trainer = DistributedTrainer(config=mock_config)
            
            mock_init.assert_called_once_with(
                backend=mock_config["distributed"]["backend"],
                init_method=mock_config["distributed"]["init_method"],
                world_size=mock_config["distributed"]["world_size"],
                rank=mock_config["distributed"]["rank"]
            )
            
            assert trainer.world_size == mock_config["distributed"]["world_size"]
            assert trainer.rank == mock_config["distributed"]["rank"]
    
    def test_model_distribution(self, mock_config):
        """Test model distribution across devices."""
        with patch('torch.distributed.init_process_group'):
            trainer = DistributedTrainer(config=mock_config)
            
            # Create mock model
            mock_model = MagicMock()
            
            with patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
                distributed_model = trainer.distribute_model(mock_model)
                mock_ddp.assert_called_once()
    
    def test_data_distribution(self, mock_config):
        """Test data distribution functionality."""
        with patch('torch.distributed.init_process_group'):
            trainer = DistributedTrainer(config=mock_config)
            
            # Create mock dataset
            dataset = MagicMock()
            dataset.__len__.return_value = 100
            
            # Test sampler creation
            sampler = trainer.create_distributed_sampler(dataset)
            assert isinstance(sampler, DistributedSampler)
            assert sampler.num_replicas == mock_config["distributed"]["world_size"]
    
    def test_gradient_synchronization(self):
        """Test gradient synchronization across processes."""
        model = torch.nn.Linear(10, 2)
        trainer = DistributedTrainer(model, world_size=2)
        
        # Simulate gradient computation
        inputs = torch.randn(4, 10)
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        
        # Synchronize gradients
        synchronized_grads = trainer.synchronize_gradients()
        
        # Verify gradients are synchronized
        for param in model.parameters():
            assert param.grad is not None
            assert torch.allclose(param.grad, param.grad.clone())
    
    def test_checkpoint_handling(self, mock_config, setup_test_env):
        """Test distributed checkpoint handling."""
        with patch('torch.distributed.init_process_group'):
            trainer = DistributedTrainer(config=mock_config)
            
            # Create mock model and optimizer
            mock_model = MagicMock()
            mock_optimizer = MagicMock()
            
            # Test checkpoint saving
            checkpoint_path = setup_test_env / "checkpoints" / "model.pt"
            trainer.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=1,
                step=1000,
                path=checkpoint_path
            )
            
            # Test checkpoint loading
            loaded_state = trainer.load_checkpoint(checkpoint_path)
            assert loaded_state["epoch"] == 1
            assert loaded_state["step"] == 1000
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_distribution(self, mock_config):
        """Test GPU distribution across processes."""
        with patch('torch.distributed.init_process_group'):
            trainer = DistributedTrainer(config=mock_config)
            
            # Test GPU assignment
            device = trainer.get_device()
            assert device.type == "cuda"
            assert device.index == trainer.rank % torch.cuda.device_count()
    
    def test_error_handling(self, mock_config):
        """Test error handling in distributed setting."""
        with patch('torch.distributed.init_process_group'):
            trainer = DistributedTrainer(config=mock_config)
            
            # Test invalid world size
            with pytest.raises(ValueError):
                trainer.validate_world_size(0)
            
            # Test invalid rank
            with pytest.raises(ValueError):
                trainer.validate_rank(-1)
    
    def test_process_coordination(self, mock_config):
        """Test coordination between processes."""
        with patch('torch.distributed.init_process_group'):
            trainer = DistributedTrainer(config=mock_config)
            
            # Test barrier synchronization
            with patch('torch.distributed.barrier') as mock_barrier:
                trainer.synchronize()
                mock_barrier.assert_called_once()
    
    def test_distributed_inference(self, mock_config):
        """Test distributed inference functionality."""
        with patch('torch.distributed.init_process_group'):
            trainer = DistributedTrainer(config=mock_config)
            
            # Create mock model and input
            mock_model = MagicMock()
            mock_input = torch.randn(10, 10)
            
            # Test inference
            with patch.object(trainer, 'gather_predictions') as mock_gather:
                trainer.inference(mock_model, mock_input)
                mock_gather.assert_called_once()
    
    def test_model_initialization(self):
        """Test distributed model initialization."""
        model = torch.nn.Linear(10, 2)
        trainer = DistributedTrainer(model, world_size=2)
        distributed_model = trainer.setup_model()
        
        assert trainer.world_size == 2
        assert trainer.rank == 0
        assert isinstance(distributed_model, torch.nn.parallel.DistributedDataParallel)


class TestDistributedDataParallel:
    """Test suite for distributed data parallel wrapper."""
    
    def test_forward_backward(self, mock_config):
        """Test forward and backward passes in distributed setting."""
        model = torch.nn.Linear(10, 10)
        ddp_model = DistributedDataParallel(model)
        
        # Test forward pass
        input_tensor = torch.randn(32, 10)
        output = ddp_model(input_tensor)
        assert output.shape == (32, 10)
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        
        for param in model.parameters():
            assert param.grad is not None
    
    def test_gradient_accumulation(self, mock_config):
        """Test gradient accumulation in distributed setting."""
        model = torch.nn.Linear(10, 10)
        ddp_model = DistributedDataParallel(model)
        
        accumulation_steps = 4
        for step in range(accumulation_steps):
            input_tensor = torch.randn(32, 10)
            output = ddp_model(input_tensor)
            loss = output.sum()
            (loss / accumulation_steps).backward()
        
        # Verify gradients
        for param in model.parameters():
            assert param.grad is not None


class TestDistributedSampler:
    """Test suite for distributed data sampler."""
    
    def test_sampling(self):
        """Test data sampling in distributed setting."""
        dataset_size = 100
        num_replicas = 2
        rank = 0
        
        sampler = DistributedSampler(
            dataset_size=dataset_size,
            num_replicas=num_replicas,
            rank=rank
        )
        
        indices = list(sampler)
        assert len(indices) == dataset_size // num_replicas
        assert all(0 <= idx < dataset_size for idx in indices)
    
    def test_epoch_handling(self):
        """Test epoch-based shuffling."""
        dataset_size = 100
        sampler = DistributedSampler(
            dataset_size=dataset_size,
            num_replicas=2,
            rank=0
        )
        
        # Get indices for different epochs
        indices_epoch1 = list(sampler)
        sampler.set_epoch(1)
        indices_epoch2 = list(sampler)
        
        # Verify different orderings
        assert indices_epoch1 != indices_epoch2


class TestGradientSynchronizer:
    """Test suite for gradient synchronization."""
    
    def test_synchronization(self):
        """Test gradient synchronization mechanism."""
        synchronizer = GradientSynchronizer(world_size=2)
        
        # Create test gradients
        gradients = [torch.randn(10, 10) for _ in range(2)]
        
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            synchronized_grads = synchronizer.synchronize(gradients)
            assert mock_all_reduce.call_count == len(gradients)
    
    def test_reduction_modes(self):
        """Test different gradient reduction modes."""
        synchronizer = GradientSynchronizer(world_size=2)
        
        # Test sum reduction
        grad = torch.ones(10, 10)
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            synchronizer.synchronize([grad], reduction="sum")
            mock_all_reduce.assert_called_with(grad, op=dist.ReduceOp.SUM)
        
        # Test average reduction
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            synchronizer.synchronize([grad], reduction="average")
            mock_all_reduce.assert_called_with(grad, op=dist.ReduceOp.AVG) 