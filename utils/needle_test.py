# src/utils/needle_test.py

"""Needle-in-haystack search test utilities.

This module provides utilities for testing search accuracy and performance
through needle-in-haystack test scenarios. These tests are designed to
verify the system's ability to find specific patterns or content within
larger datasets.
"""

import random
import string
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from utils.log_manager import LogManager, LogTemplates

logger = LogManager().get_logger(__name__)
console = Console()

def generate_haystack(
    size: int = 1000000,
    pattern_length: int = 10,
    num_needles: int = 10
) -> Tuple[str, List[str]]:
    """Generate test data with known patterns to find.
    
    Args:
        size: Total size of haystack in characters
        pattern_length: Length of each needle pattern
        num_needles: Number of needle patterns to insert
        
    Returns:
        Tuple of (haystack_text, needle_patterns)
    """
    # Generate random text
    chars = string.ascii_letters + string.digits + string.punctuation + string.whitespace
    haystack = ''.join(random.choice(chars) for _ in range(size))
    
    # Generate and insert needles
    needles = []
    for _ in range(num_needles):
        needle = ''.join(random.choice(string.ascii_letters) for _ in range(pattern_length))
        position = random.randint(0, size - pattern_length)
        haystack = haystack[:position] + needle + haystack[position + pattern_length:]
        needles.append(needle)
        
    return haystack, needles

def run_needle_tests(
    search_func: callable,
    haystack_size: int = 1000000,
    pattern_length: int = 10,
    num_needles: int = 10,
    num_trials: int = 5
) -> Dict[str, float]:
    """Run needle-in-haystack search tests.
    
    Args:
        search_func: Function that implements the search algorithm
        haystack_size: Size of test data
        pattern_length: Length of patterns to search
        num_needles: Number of patterns to search for
        num_trials: Number of test trials to run
        
    Returns:
        Dictionary with test metrics
    """
    metrics = {
        "avg_search_time": 0.0,
        "accuracy": 0.0,
        "false_positives": 0.0,
        "false_negatives": 0.0
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Running needle tests...", total=num_trials)
        
        for trial in range(num_trials):
            # Generate test data
            haystack, needles = generate_haystack(
                haystack_size,
                pattern_length,
                num_needles
            )
            
            # Run search
            start_time = time.time()
            results = search_func(haystack, needles)
            search_time = time.time() - start_time
            
            # Calculate metrics
            found_patterns = set(results)
            true_patterns = set(needles)
            
            true_positives = len(found_patterns.intersection(true_patterns))
            false_positives = len(found_patterns - true_patterns)
            false_negatives = len(true_patterns - found_patterns)
            
            metrics["avg_search_time"] += search_time
            metrics["accuracy"] += true_positives / num_needles
            metrics["false_positives"] += false_positives
            metrics["false_negatives"] += false_negatives
            
            progress.advance(task)
            
    # Average metrics
    for key in metrics:
        metrics[key] /= num_trials
        
    return metrics

def benchmark_search_algorithms(
    algorithms: Dict[str, callable],
    haystack_sizes: List[int] = [1000, 10000, 100000, 1000000],
    pattern_length: int = 10,
    num_needles: int = 10,
    num_trials: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """Benchmark multiple search algorithms.
    
    Args:
        algorithms: Dictionary mapping algorithm names to functions
        haystack_sizes: List of haystack sizes to test
        pattern_length: Length of patterns to search
        num_needles: Number of patterns per test
        num_trials: Number of trials per test
        
    Returns:
        Dictionary mapping algorithm names to their metrics
    """
    results = {name: {
        "sizes": haystack_sizes,
        "times": [],
        "accuracies": []
    } for name in algorithms}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task(
            "Benchmarking algorithms...",
            total=len(algorithms) * len(haystack_sizes)
        )
        
        for name, func in algorithms.items():
            for size in haystack_sizes:
                metrics = run_needle_tests(
                    func,
                    haystack_size=size,
                    pattern_length=pattern_length,
                    num_needles=num_needles,
                    num_trials=num_trials
                )
                
                results[name]["times"].append(metrics["avg_search_time"])
                results[name]["accuracies"].append(metrics["accuracy"])
                
                progress.advance(task)
                
    return results

if __name__ == "__main__":
    # Example usage
    def simple_search(haystack: str, needles: List[str]) -> List[str]:
        """Simple string search implementation."""
        return [needle for needle in needles if needle in haystack]
        
    metrics = run_needle_tests(simple_search)
    console.print("Test Results:")
    for key, value in metrics.items():
        console.print(f"{key}: {value:.4f}")
