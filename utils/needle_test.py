# src/utils/needle_test.py

"""Needle-in-haystack testing utilities for LlamaHome.

This module provides tools for testing model performance on rare but important patterns
within larger datasets. It helps validate model sensitivity and accuracy for edge cases.
"""

import random
from typing import List, Dict, Any
from utils.log_manager import LogManager, LogTemplates

class NeedleTest:
    """Implements needle-in-haystack testing methodology."""

    def __init__(self):
        """Initialize needle test with logging."""
        self.logger = LogManager().get_logger("needle_test", "system", "monitor")
        
    def generate_haystack(self, size: int = 1000, needle_count: int = 1) -> Dict[str, Any]:
        """Generate test data with rare patterns embedded.
        
        Args:
            size: Size of the dataset to generate
            needle_count: Number of needle patterns to embed
            
        Returns:
            Dictionary containing generated data and needle information
        """
        haystack = []
        needles = []
        
        # Generate needle patterns (rare but important cases)
        for _ in range(needle_count):
            needle = {
                "type": "needle",
                "pattern": f"rare_pattern_{random.randint(1000, 9999)}",
                "priority": "high",
                "expected_response": "specific_handling_required"
            }
            needles.append(needle)
            
        # Generate regular data
        for i in range(size):
            haystack.append({
                "type": "regular",
                "pattern": f"common_pattern_{i}",
                "priority": "normal",
                "expected_response": "standard_handling"
            })
            
        # Insert needles at random positions
        for needle in needles:
            position = random.randint(0, len(haystack))
            haystack.insert(position, needle)
            
        return {
            "data": haystack,
            "needles": needles
        }

    def run_needle_test(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run needle-in-haystack test on model.
        
        Args:
            model: Model instance to test
            test_data: Dictionary containing test data and needle information
            
        Returns:
            Dictionary containing test results and metrics
            
        Raises:
            Exception: If test execution fails
        """
        results = {
            "total_samples": len(test_data["data"]),
            "needle_count": len(test_data["needles"]),
            "found_needles": 0,
            "false_positives": 0,
            "missed_needles": 0
        }
        
        try:
            self.logger.info(LogTemplates.SYSTEM_INFO.format(
                info=f"Starting needle test with {results['needle_count']} needles in {results['total_samples']} samples"
            ))
            
            needle_patterns = {n["pattern"] for n in test_data["needles"]}
            
            for item in test_data["data"]:
                response = model.process(item)
                
                if item["type"] == "needle":
                    if response["detected_pattern"] in needle_patterns:
                        results["found_needles"] += 1
                    else:
                        results["missed_needles"] += 1
                else:
                    if response["detected_pattern"] in needle_patterns:
                        results["false_positives"] += 1
                        
            self.logger.info(LogTemplates.SYSTEM_SUCCESS.format(
                success=f"Needle test completed: {results['found_needles']}/{results['needle_count']} needles found"
            ))
            
            return results
            
        except Exception as e:
            self.logger.error(LogTemplates.SYSTEM_ERROR.format(
                error=f"Needle test failed: {str(e)}"
            ))
            raise

def run_needle_tests(model, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a suite of needle tests.
    
    Args:
        model: Model instance to test
        config: Optional configuration dictionary for test parameters
        
    Returns:
        Dictionary containing aggregated test results
        
    Example:
        >>> config = {
        ...     "haystack_size": 2000,
        ...     "needle_count": 5,
        ...     "test_iterations": 10
        ... }
        >>> results = run_needle_tests(model, config)
        >>> print(f"Detection Rate: {results['total_found'] / results['total_needles']:.2%}")
    """
    if config is None:
        config = {
            "haystack_size": 1000,
            "needle_count": 3,
            "test_iterations": 5
        }
        
    tester = NeedleTest()
    aggregated_results = {
        "total_tests": config["test_iterations"],
        "total_needles": 0,
        "total_found": 0,
        "total_missed": 0,
        "false_positive_rate": 0.0,
        "individual_results": []
    }
    
    for i in range(config["test_iterations"]):
        test_data = tester.generate_haystack(
            size=config["haystack_size"],
            needle_count=config["needle_count"]
        )
        result = tester.run_needle_test(model, test_data)
        aggregated_results["individual_results"].append(result)
        
        aggregated_results["total_needles"] += result["needle_count"]
        aggregated_results["total_found"] += result["found_needles"]
        aggregated_results["total_missed"] += result["missed_needles"]
        
    # Calculate final metrics
    aggregated_results["false_positive_rate"] = sum(
        r["false_positives"] / r["total_samples"] 
        for r in aggregated_results["individual_results"]
    ) / config["test_iterations"]
    
    return aggregated_results

# Example usage
if __name__ == "__main__":
    # This section demonstrates how to use the needle tests
    from rich.console import Console
    console = Console()
    
    console.print("[bold blue]Running needle tests demonstration...[/bold blue]")
    
    # Mock model for demonstration
    class MockModel:
        def process(self, item):
            return {"detected_pattern": item["pattern"]}
    
    model = MockModel()
    
    config = {
        "haystack_size": 2000,
        "needle_count": 5,
        "test_iterations": 10
    }
    
    results = run_needle_tests(model, config)
    
    console.print("\n[bold green]Test Results:[/bold green]")
    console.print(f"Total Tests: {results['total_tests']}")
    console.print(f"Total Needles: {results['total_needles']}")
    console.print(f"Found Needles: {results['total_found']}")
    console.print(f"Detection Rate: {results['total_found'] / results['total_needles']:.2%}")
    console.print(f"False Positive Rate: {results['false_positive_rate']:.2%}")
