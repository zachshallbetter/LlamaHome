"""Benchmarking utilities for performance testing."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..utils import LogManager, LogTemplates


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    name: str
    duration: float
    memory_used: Optional[float] = None
    metrics: Dict[str, float] = None


class BenchmarkManager:
    """Manages performance benchmarking."""

    def __init__(self):
        """Initialize benchmark manager."""
        self.logger = LogManager(LogTemplates.BENCHMARK).get_logger(__name__)
        self.results: List[BenchmarkResult] = []

    def start_benchmark(self, name: str) -> float:
        """Start timing a benchmark.
        
        Args:
            name: Name of the benchmark
            
        Returns:
            Start time in seconds
        """
        self.logger.info(f"Starting benchmark: {name}")
        return time.time()

    def end_benchmark(self, name: str, start_time: float, metrics: Dict[str, float] = None) -> BenchmarkResult:
        """End timing a benchmark.
        
        Args:
            name: Name of the benchmark
            start_time: Start time from start_benchmark
            metrics: Optional metrics to record
            
        Returns:
            Benchmark result
        """
        duration = time.time() - start_time
        result = BenchmarkResult(name=name, duration=duration, metrics=metrics or {})
        self.results.append(result)
        self.logger.info(f"Benchmark {name} completed in {duration:.2f}s")
        return result

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results.
        
        Returns:
            List of benchmark results
        """
        return self.results

    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self.results.clear()
        self.logger.info("Cleared benchmark results") 