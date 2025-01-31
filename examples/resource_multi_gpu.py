"""
Example script demonstrating multi-GPU resource management in LlamaHome.
"""

import asyncio
from pathlib import Path

import torch
from src.core import GPUConfig, ResourceConfig, MultiGPUManager, DeviceAllocator


async def main():
    """Run multi-GPU management example."""
    # Check GPU availability
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This example requires at least 2 GPUs")

    # Configuration
    config = ResourceConfig(
        gpu=GPUConfig(
            memory_fraction=0.8,
            priority_devices=[0, 1],  # Prioritize first two GPUs
            enable_peer_access=True,
            balance_load=True,
        )
    )

    # Initialize managers
    manager = MultiGPUManager(config)
    allocator = DeviceAllocator(manager)

    print("Starting multi-GPU management example...")

    try:
        # Get GPU information
        print("\nAvailable GPUs:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

        # Demonstrate GPU allocation
        print("\nAllocating workloads...")

        async with manager.distribute() as devices:
            # Simulate distributed workload
            tensors = []
            for i, device in enumerate(devices):
                # Create tensor on each GPU
                size = 2000 * (i + 1)  # Different sizes for demonstration
                tensor = torch.randn(size, size, device=device)
                tensors.append(tensor)

                # Get device stats
                stats = await manager.get_device_stats(device)
                print(f"\nDevice {device} stats:")
                print(f"Memory Used: {stats['memory_used'] / 1024**3:.1f} GB")
                print(f"Memory Free: {stats['memory_free'] / 1024**3:.1f} GB")
                print(f"Utilization: {stats['utilization']:.1%}")

            # Demonstrate peer access
            if config.gpu.enable_peer_access:
                print("\nTesting peer access...")
                for i, tensor1 in enumerate(tensors):
                    for j, tensor2 in enumerate(tensors):
                        if i != j:
                            # Transfer data between GPUs
                            result = torch.matmul(tensor1, tensor2.to(tensor1.device))
                            print(f"Transfer {i}->{j} shape: {result.shape}")

        print("\nFinal GPU states:")
        await manager.print_gpu_states()

    except Exception as e:
        print(f"Multi-GPU management failed: {e}")
        raise


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
