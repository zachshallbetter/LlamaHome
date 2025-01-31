"""
Example script demonstrating performance monitoring in LlamaHome.
"""

import asyncio
from pathlib import Path

import torch
from src.core import MonitorConfig, ResourceConfig, ResourceManager, PerformanceMonitor


async def main():
    """Run performance monitoring example."""
    # Configuration
    config = MonitorConfig(
        metrics=["gpu", "cpu", "memory", "io"],
        interval=1.0,  # seconds
        history_size=100,
        alert_thresholds={
            "gpu_usage": 0.95,
            "cpu_usage": 0.90,
            "memory_usage": 0.85,
            "io_queue": 1000,
        },
    )

    # Initialize monitor
    monitor = PerformanceMonitor(config)

    print("Starting performance monitoring...")

    try:
        # Start monitoring
        await monitor.start()

        # Simulate workload
        print("\nSimulating workload...")
        for i in range(5):
            # Create some GPU load
            tensor = torch.randn(5000, 5000, device="cuda")
            torch.matmul(tensor, tensor.T)

            # Get current metrics
            metrics = await monitor.get_metrics()
            print(f"\nIteration {i+1} metrics:")
            print(f"GPU Usage: {metrics['gpu_usage']:.2%}")
            print(f"Memory Usage: {metrics['memory_usage']:.2%}")
            print(f"CPU Usage: {metrics['cpu_usage']:.2%}")

            await asyncio.sleep(1)

        # Get performance summary
        print("\nPerformance Summary:")
        summary = await monitor.get_summary()
        print(f"Peak GPU Usage: {summary['peak_gpu_usage']:.2%}")
        print(f"Average Memory Usage: {summary['avg_memory_usage']:.2%}")
        print(
            f"CPU Usage Range: {summary['min_cpu_usage']:.2%} - {summary['max_cpu_usage']:.2%}"
        )

    except Exception as e:
        print(f"Monitoring failed: {e}")
        raise

    finally:
        # Stop monitoring
        await monitor.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
