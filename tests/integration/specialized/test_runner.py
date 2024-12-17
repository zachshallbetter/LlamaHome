"""Specialized test runner for LlamaHome.

This module provides a test runner specifically designed for specialized
test cases, including needle-in-haystack tests, edge cases, and stress tests.
It integrates with the main test infrastructure while providing additional
functionality for specialized test scenarios.
"""

import os
import sys
import toml
import pytest
from pathlib import Path
from typing import Dict, List, Optional, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils.log_manager import LogManager, LogTemplates
from src.testing.needle_test import run_needle_tests, benchmark_search_algorithms
from utils.system_check import check_resources

console = Console()
logger = LogManager().get_logger(__name__)

class SpecializedTestRunner:
    """Runner for specialized test cases."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize specialized test runner.
        
        Args:
            config_path: Path to test configuration file
        """
        self.config = self._load_config(config_path)
        self.console = Console()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load test configuration."""
        if config_path is None:
            config_path = Path("tests/test_config.toml")
            
        try:
            with open(config_path) as f:
                return toml.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
            
    def _check_requirements(self, category: str) -> bool:
        """Check if system meets test requirements."""
        if category not in self.config["categories"]:
            return False
            
        category_config = self.config["categories"][category]
        if not category_config.get("enabled", False):
            return False
            
        if "requirements" in category_config:
            return check_resources(category_config["requirements"])
            
        return True
        
    def run_needle_tests(self, **kwargs) -> Dict[str, Any]:
        """Run needle-in-haystack tests."""
        if not self._check_requirements("specialized"):
            logger.warning("Specialized tests are disabled or requirements not met")
            return {}
            
        try:
            # Get test configuration
            needle_config = self.config["test_data"]["datasets"]["needle_search"]
            
            # Run tests for each configuration
            results = {}
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}")
            ) as progress:
                task = progress.add_task(
                    "Running needle tests...",
                    total=len(needle_config["pattern_lengths"]) * len(needle_config["needle_counts"])
                )
                
                for pattern_length in needle_config["pattern_lengths"]:
                    for needle_count in needle_config["needle_counts"]:
                        test_key = f"p{pattern_length}_n{needle_count}"
                        
                        # Run test with current configuration
                        metrics = run_needle_tests(
                            search_func=kwargs.get("search_func"),
                            pattern_length=pattern_length,
                            num_needles=needle_count,
                            **kwargs
                        )
                        
                        results[test_key] = metrics
                        progress.advance(task)
                        
            return results
            
        except Exception as e:
            logger.error(f"Needle tests failed: {e}")
            return {}
            
    def run_edge_case_tests(self) -> Dict[str, Any]:
        """Run edge case tests."""
        if not self._check_requirements("specialized"):
            return {}
            
        # TODO: Implement edge case tests
        return {}
        
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests."""
        if not self._check_requirements("specialized"):
            return {}
            
        # TODO: Implement stress tests
        return {}
        
    def run_all_specialized_tests(self, **kwargs) -> Dict[str, Any]:
        """Run all specialized tests."""
        results = {
            "needle_tests": self.run_needle_tests(**kwargs),
            "edge_case_tests": self.run_edge_case_tests(),
            "stress_tests": self.run_stress_tests()
        }
        
        # Generate report
        self._generate_report(results)
        
        return results
        
    def _generate_report(self, results: Dict[str, Any]):
        """Generate test report."""
        try:
            report_dir = Path(self.config["environment"]["artifacts_dir"])
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = report_dir / "specialized_test_report.toml"
            with open(report_path, "w") as f:
                toml.dump(results, f, default_flow_style=False)
                
            self.console.print(f"[green]Report generated: {report_path}[/green]")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

def main():
    """Main entry point for specialized test runner."""
    try:
        runner = SpecializedTestRunner()
        
        # Example search function for demonstration
        def example_search(haystack: str, needles: List[str]) -> List[str]:
            return [needle for needle in needles if needle in haystack]
            
        results = runner.run_all_specialized_tests(search_func=example_search)
        
        if results["needle_tests"]:
            console.print("\n[bold green]Needle Test Results:[/bold green]")
            for test_key, metrics in results["needle_tests"].items():
                console.print(f"\nTest Configuration: {test_key}")
                for metric, value in metrics.items():
                    console.print(f"{metric}: {value:.4f}")
                    
    except Exception as e:
        console.print(f"[red]Error running specialized tests: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 