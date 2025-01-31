"""Example of multi-GPU resource management."""

import asyncio
from typing import Dict

import torch

from src.core.resource import GPUConfig, MultiGPUManager


async def manage_multi_gpu() -> None:
    """Demonstrate multi-GPU resource management."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Configure GPU resources
    gpu_config = GPUConfig(
        memory_fraction=0.8,
        allow_growth=True,
        allowed_devices=[0, 1],  # Use first two GPUs
    )

    # Initialize manager
    manager = MultiGPUManager(gpu_config)

    try:
        # Allocate resources
        async with manager.optimize():
            # Simulate workload
            for _i in range(5):
                memory_info = await manager.get_memory_info()
                print("\nGPU Memory Usage:")
                for device, usage in memory_info.items():
                    print(f"GPU {device}: {usage:.1f}GB")
                await asyncio.sleep(1)

    except Exception as e:
        print(f"Resource management failed: {e}")


if __name__ == "__main__":
    asyncio.run(manage_multi_gpu())
