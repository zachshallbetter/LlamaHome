"""Benchmark utilities for performance measurement."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.log_manager import LogManager, LogTemplates, Singleton

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result container."""

    name: str
    duration: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class BenchmarkManager(metaclass=Singleton):
    """Manage benchmarking operations."""

    _initialized: bool = False

    def __init__(self):
        """Initialize benchmark manager."""
        if not BenchmarkManager._initialized:
            logger.info("Initializing benchmark manager")
            self.results: List[BenchmarkResult] = []
            self._setup_metrics()
            BenchmarkManager._initialized = True

    def _setup_metrics(self) -> None:
        """Set up metrics collection."""
        try:
            import psutil
            self.process = psutil.Process()
            logger.debug("Successfully initialized psutil metrics")
        except ImportError:
            logger.warning("psutil not available, some metrics will be limited")
            self.process = None

    def clear_results(self) -> None:
        """Clear stored benchmark results."""
        logger.debug("Clearing benchmark results")
        self.results.clear()

    @contextmanager
    def measure(self, name: str) -> None:
        """Context manager for measuring execution.

        Args:
            name: Name of the benchmark
        """
        logger.debug(f"Starting benchmark measurement: {name}")
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()

        try:
            yield
        finally:
            duration = time.time() - start_time
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()

            memory_usage = end_memory - start_memory if all([end_memory, start_memory]) else None
            cpu_usage = end_cpu - start_cpu if all([end_cpu, start_cpu]) else None

            result = BenchmarkResult(
                name=name, duration=duration, memory_usage=memory_usage, cpu_usage=cpu_usage
            )
            self.results.append(result)
            self._log_result(result)
            logger.debug(f"Completed benchmark measurement: {name}")

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage.

        Returns:
            Memory usage in MB if available
        """
        if self.process:
            try:
                return self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
            except Exception as e:
                logger.debug(f"Failed to get memory usage: {e}")
        return None

    def _get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage.

        Returns:
            CPU usage percentage if available
        """
        if self.process:
            try:
                return self.process.cpu_percent()
            except Exception as e:
                logger.debug(f"Failed to get CPU usage: {e}")
        return None

    def _log_result(self, result: BenchmarkResult) -> None:
        """Log benchmark result.

        Args:
            result: Benchmark result to log
        """
        message = f"Benchmark {result.name}: {result.duration:.3f}s"
        if result.memory_usage is not None:
            message += f", Memory: {result.memory_usage:.2f}MB"
        if result.cpu_usage is not None:
            message += f", CPU: {result.cpu_usage:.1f}%"
        logger.info(message)

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results.

        Returns:
            List of benchmark results
        """
        logger.debug("Retrieving benchmark results")
        return self.results.copy()

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of benchmark results.

        Returns:
            Dictionary of benchmark summaries
        """
        logger.debug("Generating benchmark summary")
        summary = {}
        for result in self.results:
            metrics = {"duration": result.duration}
            if result.memory_usage is not None:
                metrics["memory_usage"] = result.memory_usage
            if result.cpu_usage is not None:
                metrics["cpu_usage"] = result.cpu_usage
            if result.additional_metrics:
                metrics.update(result.additional_metrics)
            summary[result.name] = metrics
        return summary


# Global benchmark manager instance
logger.info("Creating global benchmark manager instance")
benchmark_manager = BenchmarkManager()


@contextmanager
def benchmark(name: str) -> None:
    """Convenience context manager for benchmarking.

    Args:
        name: Name of the benchmark
    """
    logger.debug(f"Using benchmark context manager: {name}")
    with benchmark_manager.measure(name):
        yield


def clear_benchmarks() -> None:
    """Clear all benchmark results."""
    logger.info("Clearing all benchmark results")
    benchmark_manager.clear_results()


def get_benchmark_results() -> List[BenchmarkResult]:
    """Get all benchmark results.

    Returns:
        List of benchmark results
    """
    logger.debug("Retrieving all benchmark results")
    return benchmark_manager.get_results()


def get_benchmark_summary() -> Dict[str, Dict[str, float]]:
    """Get summary of all benchmarks.

    Returns:
        Dictionary of benchmark summaries
    """
    logger.debug("Retrieving benchmark summary")
    return benchmark_manager.get_summary()
