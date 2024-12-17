"""
Test result reporting and visualization system.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

class TestReportGenerator:
    def __init__(self, output_dir: str = "test-results"):
        """Initialize the report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        
    def process_test_results(self, results_file: Path) -> Dict:
        """Process raw test results into a structured format."""
        with open(results_file) as f:
            results = json.load(f)
            
        processed_results = {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r["status"] == "passed"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "duration": sum(r["duration"] for r in results),
            "timestamp": datetime.now().isoformat(),
            "details": results
        }
        return processed_results
        
    def generate_visualizations(self, results: Dict) -> List[str]:
        """Generate visualization plots for test results."""
        plot_files = []
        
        # Test status distribution pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(
            [results["passed"], results["failed"], results["skipped"]],
            labels=["Passed", "Failed", "Skipped"],
            autopct="%1.1f%%",
            colors=["#28a745", "#dc3545", "#ffc107"]
        )
        plt.title("Test Results Distribution")
        pie_chart_path = self.output_dir / "test_distribution.png"
        plt.savefig(pie_chart_path)
        plt.close()
        plot_files.append(str(pie_chart_path))
        
        # Test duration bar chart
        durations = [r["duration"] for r in results["details"]]
        plt.figure(figsize=(12, 6))
        plt.hist(durations, bins=30)
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Number of Tests")
        plt.title("Test Duration Distribution")
        duration_chart_path = self.output_dir / "test_durations.png"
        plt.savefig(duration_chart_path)
        plt.close()
        plot_files.append(str(duration_chart_path))
        
        return plot_files
        
    def generate_html_report(self, results: Dict, plot_files: List[str]) -> str:
        """Generate HTML report from test results and plots."""
        template = self.env.get_template("report_template.html")
        report_content = template.render(
            results=results,
            plots=plot_files,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        report_path = self.output_dir / "test_report.html"
        with open(report_path, "w") as f:
            f.write(report_content)
            
        return str(report_path)
        
    def generate_report(self, results_file: Path) -> str:
        """Generate complete test report including visualizations."""
        results = self.process_test_results(results_file)
        plot_files = self.generate_visualizations(results)
        report_path = self.generate_html_report(results, plot_files)
        
        # Generate summary JSON
        summary = {
            "total_tests": results["total_tests"],
            "passed": results["passed"],
            "failed": results["failed"],
            "skipped": results["skipped"],
            "duration": results["duration"],
            "report_path": report_path
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        return report_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test report")
    parser.add_argument("results_file", type=Path, help="Path to test results JSON file")
    parser.add_argument("--output-dir", type=str, default="test-results",
                      help="Output directory for report files")
    
    args = parser.parse_args()
    
    generator = TestReportGenerator(output_dir=args.output_dir)
    report_path = generator.generate_report(args.results_file)
    print(f"Report generated: {report_path}") 