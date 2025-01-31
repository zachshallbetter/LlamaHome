"""
Example script demonstrating memory management in LlamaHome.
"""

import asyncio

from src.core.resource.config import MemoryConfig, MonitorConfig
from src.core.resource.monitor import PerformanceMonitor


async def manage_memory() -> None:
    """Demonstrate memory management."""
    # Configure memory management
    memory_config = MemoryConfig(
        cache_size="4GB", min_free="2GB", cleanup_margin=0.1, check_interval=1.0
    )

    # Initialize monitor
    monitor = PerformanceMonitor(MonitorConfig())

    try:
        while True:
            # Check memory usage
            metrics = await monitor.check_resources()
            memory_percent = metrics["memory_percent"]

            print(f"\nMemory Usage: {memory_percent:.1f}%")

            # Check if cleanup needed
            if memory_percent > (1 - memory_config.cleanup_margin) * 100:
                print("Memory cleanup recommended!")

            await asyncio.sleep(memory_config.check_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    asyncio.run(manage_memory())
