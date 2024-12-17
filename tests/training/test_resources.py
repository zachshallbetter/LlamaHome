"""Tests for training resource management system."""

import pytest
import torch
import psutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.resources import (
    ResourceManager,
    GPUManager,
    CPUManager,
    DiskManager,
    ResourceMonitor
)


@pytest.fixture
def mock_config():
    """Create mock resource configuration."""
    return {
        "resources": {
            "gpu": {
                "memory_fraction": 0.9,
                "allow_growth": True,
                "per_process_memory": "12GB",
                "enable_tf32": True
            },
            "cpu": {
                "num_workers": 4,
                "pin_memory": True,
                "affinity": "performance",
                "thread_pool_size": 8
            },
            "disk": {
                "cache_dir": ".cache",
                "min_free_space": "10GB",
                "cleanup_threshold": 0.9
            },
            "monitoring": {
                "interval": 1.0,
                "metrics": ["memory", "utilization", "temperature"],
                "log_to_file": True
            }
        }
    }


@pytest.fixture
def setup_test_env(tmp_path):
    """Set up test environment."""
    # Create resource directories
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True)
    
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    
    return tmp_path


class TestResourceManager:
    """Test suite for main resource management functionality."""
    
    def test_initialization(self, mock_config, setup_test_env):
        """Test resource manager initialization."""
        manager = ResourceManager(
            config=mock_config,
            workspace_dir=setup_test_env
        )
        
        assert manager.gpu_enabled == torch.cuda.is_available()
        assert manager.num_workers == mock_config["resources"]["cpu"]["num_workers"]
        assert manager.cache_dir == setup_test_env / ".cache"
    
    def test_resource_allocation(self, mock_config, setup_test_env):
        """Test resource allocation functionality."""
        manager = ResourceManager(
            config=mock_config,
            workspace_dir=setup_test_env
        )
        
        # Test resource allocation
        with manager.allocate_resources() as resources:
            assert "gpu" in resources or not torch.cuda.is_available()
            assert "cpu" in resources
            assert "disk" in resources
    
    def test_resource_limits(self, mock_config, setup_test_env):
        """Test resource limit enforcement."""
        manager = ResourceManager(
            config=mock_config,
            workspace_dir=setup_test_env
        )
        
        # Test memory limits
        with pytest.raises(RuntimeError):
            with manager.enforce_limits():
                # Try to allocate too much memory
                huge_tensor = torch.randn(100000, 100000)
    
    def test_resource_cleanup(self, mock_config, setup_test_env):
        """Test resource cleanup functionality."""
        manager = ResourceManager(
            config=mock_config,
            workspace_dir=setup_test_env
        )
        
        # Allocate resources
        tensors = [torch.randn(1000, 1000) for _ in range(10)]
        
        # Cleanup
        manager.cleanup()
        
        # Verify cleanup
        if torch.cuda.is_available():
            assert torch.cuda.memory_allocated() == 0


class TestGPUManager:
    """Test suite for GPU management functionality."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_gpu_initialization(self, mock_config):
        """Test GPU manager initialization."""
        gpu_manager = GPUManager(config=mock_config["resources"]["gpu"])
        
        assert gpu_manager.memory_fraction == 0.9
        assert gpu_manager.allow_growth
        assert gpu_manager.enable_tf32
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_memory_management(self, mock_config):
        """Test GPU memory management."""
        gpu_manager = GPUManager(config=mock_config["resources"]["gpu"])
        
        # Test memory allocation
        with gpu_manager.memory_scope():
            tensor = torch.randn(1000, 1000, device="cuda")
            assert tensor.device.type == "cuda"
            
            # Check memory tracking
            stats = gpu_manager.get_memory_stats()
            assert stats["allocated"] > 0
            assert stats["cached"] >= stats["allocated"]
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_multi_gpu(self, mock_config):
        """Test multi-GPU functionality."""
        gpu_manager = GPUManager(config=mock_config["resources"]["gpu"])
        
        if torch.cuda.device_count() > 1:
            # Test device assignment
            devices = gpu_manager.get_available_devices()
            assert len(devices) > 1
            
            # Test memory stats for each device
            for device in devices:
                stats = gpu_manager.get_device_stats(device)
                assert "memory" in stats
                assert "utilization" in stats
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_optimization_features(self, mock_config):
        """Test GPU optimization features."""
        gpu_manager = GPUManager(config=mock_config["resources"]["gpu"])
        
        # Test TF32 settings
        gpu_manager.configure_precision()
        assert torch.backends.cuda.matmul.allow_tf32 == gpu_manager.enable_tf32
        
        # Test memory optimization
        with gpu_manager.optimize_memory():
            tensor = torch.randn(1000, 1000, device="cuda")
            del tensor
            torch.cuda.empty_cache()
            assert torch.cuda.memory_allocated() == 0


class TestCPUManager:
    """Test suite for CPU management functionality."""
    
    def test_cpu_initialization(self, mock_config):
        """Test CPU manager initialization."""
        cpu_manager = CPUManager(config=mock_config["resources"]["cpu"])
        
        assert cpu_manager.num_workers == 4
        assert cpu_manager.pin_memory
        assert cpu_manager.thread_pool_size == 8
    
    def test_thread_management(self, mock_config):
        """Test CPU thread management."""
        cpu_manager = CPUManager(config=mock_config["resources"]["cpu"])
        
        # Test thread pool
        with cpu_manager.thread_pool() as pool:
            assert pool._max_workers == cpu_manager.thread_pool_size
            
            # Run some parallel tasks
            results = list(pool.map(lambda x: x*x, range(10)))
            assert results == [x*x for x in range(10)]
    
    def test_affinity_management(self, mock_config):
        """Test CPU affinity management."""
        cpu_manager = CPUManager(config=mock_config["resources"]["cpu"])
        
        # Test affinity settings
        with cpu_manager.set_affinity("performance"):
            process = psutil.Process()
            assert len(process.cpu_affinity()) > 0
    
    def test_memory_management(self, mock_config):
        """Test CPU memory management."""
        cpu_manager = CPUManager(config=mock_config["resources"]["cpu"])
        
        # Test memory pinning
        with cpu_manager.pin_memory():
            tensor = torch.randn(1000, 1000)
            assert tensor.is_pinned() == cpu_manager.pin_memory


class TestDiskManager:
    """Test suite for disk management functionality."""
    
    def test_disk_initialization(self, mock_config, setup_test_env):
        """Test disk manager initialization."""
        disk_manager = DiskManager(
            config=mock_config["resources"]["disk"],
            workspace_dir=setup_test_env
        )
        
        assert disk_manager.cache_dir == setup_test_env / ".cache"
        assert disk_manager.min_free_space == "10GB"
    
    def test_space_management(self, mock_config, setup_test_env):
        """Test disk space management."""
        disk_manager = DiskManager(
            config=mock_config["resources"]["disk"],
            workspace_dir=setup_test_env
        )
        
        # Test space checking
        free_space = disk_manager.get_free_space()
        assert free_space > 0
        
        # Test space requirement checking
        assert disk_manager.has_sufficient_space("1GB")
    
    def test_cleanup(self, mock_config, setup_test_env):
        """Test disk cleanup functionality."""
        disk_manager = DiskManager(
            config=mock_config["resources"]["disk"],
            workspace_dir=setup_test_env
        )
        
        # Create some test files
        for i in range(10):
            with open(disk_manager.cache_dir / f"test_{i}.txt", "w") as f:
                f.write("x" * 1000000)  # 1MB file
        
        # Trigger cleanup
        disk_manager.cleanup()
        
        # Verify cleanup
        remaining_files = list(disk_manager.cache_dir.glob("*"))
        assert len(remaining_files) < 10
    
    def test_file_management(self, mock_config, setup_test_env):
        """Test file management functionality."""
        disk_manager = DiskManager(
            config=mock_config["resources"]["disk"],
            workspace_dir=setup_test_env
        )
        
        # Test file operations
        test_file = disk_manager.cache_dir / "test.txt"
        disk_manager.save_file(test_file, "test content")
        assert test_file.exists()
        
        content = disk_manager.load_file(test_file)
        assert content == "test content"
        
        disk_manager.delete_file(test_file)
        assert not test_file.exists()


class TestResourceMonitor:
    """Test suite for resource monitoring functionality."""
    
    def test_monitor_initialization(self, mock_config, setup_test_env):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor(
            config=mock_config["resources"]["monitoring"],
            log_dir=setup_test_env / "logs"
        )
        
        assert monitor.interval == 1.0
        assert "memory" in monitor.metrics
        assert monitor.log_to_file
    
    def test_metric_collection(self, mock_config, setup_test_env):
        """Test metric collection functionality."""
        monitor = ResourceMonitor(
            config=mock_config["resources"]["monitoring"],
            log_dir=setup_test_env / "logs"
        )
        
        # Collect metrics
        metrics = monitor.collect_metrics()
        
        assert "cpu" in metrics
        assert "memory" in metrics
        if torch.cuda.is_available():
            assert "gpu" in metrics
    
    def test_monitoring_session(self, mock_config, setup_test_env):
        """Test monitoring session functionality."""
        monitor = ResourceMonitor(
            config=mock_config["resources"]["monitoring"],
            log_dir=setup_test_env / "logs"
        )
        
        # Run monitoring session
        with monitor.session("test_session"):
            # Simulate some work
            tensor = torch.randn(1000, 1000)
            del tensor
        
        # Verify logs
        assert (setup_test_env / "logs" / "test_session.log").exists()
    
    def test_alert_system(self, mock_config, setup_test_env):
        """Test resource monitoring alert system."""
        monitor = ResourceMonitor(
            config=mock_config["resources"]["monitoring"],
            log_dir=setup_test_env / "logs"
        )
        
        # Register alert handler
        alerts_received = []
        def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.register_alert_handler(alert_handler)
        
        # Trigger alert condition
        with monitor.session("test_alerts"):
            monitor.check_alert_conditions({
                "memory_usage": 0.95,  # High memory usage
                "gpu_temperature": 85   # High temperature
            })
        
        assert len(alerts_received) > 0 