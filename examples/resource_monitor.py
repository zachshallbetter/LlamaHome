"""
Example script demonstrating performance monitoring in LlamaHome.
"""

import asyncio

from src.core.resource.config import MonitorConfig
from src.core.resource.monitor import PerformanceMonitor


async def monitor_resources() -> None:
    """Monitor system resources."""
    # Configure monitoring
    monitor_config = MonitorConfig(
        check_interval=1.0,
        memory_threshold=0.8,
        cpu_threshold=0.7,
        gpu_temp_threshold=75.0,
    )

    # Initialize monitor
    monitor = PerformanceMonitor(monitor_config)

    try:
        while True:
            # Check resources
            metrics = await monitor.check_resources()

            # Print metrics
            print("\nResource Usage:")
            for name, value in metrics.items():
                print(f"{name}: {value:.1f}%")

            # Check if optimization needed
            if await monitor.should_optimize():
                print("\nResource optimization recommended!")

            await asyncio.sleep(monitor_config.check_interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    asyncio.run(monitor_resources())
