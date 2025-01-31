"""
Example script demonstrating memory management in LlamaHome.
"""

import asyncio
from pathlib import Path

import torch
from src.core import MemoryConfig, ResourceConfig, ResourceManager, MemoryTracker


async def main():
    """Run memory management example."""
    # Configuration
    config = ResourceConfig(
        memory=MemoryConfig(
            gpu_reserved=1024,  # MB
            cpu_reserved=2048,  # MB
            swap_threshold=0.8,
            cleanup_threshold=0.9,
        ),
        gpu_memory_fraction=0.9,
        cpu_usage_threshold=0.8,
        io_queue_size=1000,
    )

    # Initialize resource manager
    manager = ResourceManager(config)
    tracker = MemoryTracker()

    print("Starting memory management example...")

    try:
        # Monitor initial state
        print("\nInitial memory state:")
        print(tracker.get_memory_stats())

        # Demonstrate memory optimization
        with manager.optimize():
            # Simulate memory-intensive operation
            large_tensor = torch.randn(1000, 1000, 1000)
            print("\nAfter allocation:")
            print(tracker.get_memory_stats())

            # Force garbage collection
            del large_tensor
            await manager.cleanup()
            print("\nAfter cleanup:")
            print(tracker.get_memory_stats())

        print("\nFinal memory state:")
        print(tracker.get_memory_stats())

    except Exception as e:
        print(f"Memory management failed: {e}")
        raise


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
