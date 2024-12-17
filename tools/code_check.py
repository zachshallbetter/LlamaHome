"""Code quality check utilities."""

import concurrent.futures
import sys
from pathlib import Path
from typing import Dict, List, Optional
import toml
from rich.console import Console
from rich.table import Table

console = Console()

class CodeChecker:
    """Code quality checker."""
    
    def __init__(self):
        """Initialize code checker."""
        self.workspace_root = Path.cwd()
        self.config_file = self.workspace_root / ".config/code_check.toml"
        self.load_config()
        
    def load_config(self) -> None:
        """Load code check configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = toml.load(f)
        else:
            raise FileNotFoundError(f"Code check configuration not found: {self.config_file}")
            
    def run_checks(self, files: Optional[List[str]] = None) -> bool:
        """Run code quality checks.
        
        Args:
            files: Optional list of files to check
            
        Returns:
            True if all checks pass
        """
        # Get execution config
        jobs = self.config["execution"].get("jobs", 4)
        use_cache = self.config["execution"].get("cache", True)
        
        # Get reporting config
        show_summary = self.config["reporting"].get("summary", True)
        show_details = self.config["reporting"].get("details", True)
        save_results = self.config["reporting"].get("save-to-file", "code_check_results.txt")
        
        # Collect Python files, respecting ignore patterns
        if files is None:
            files = []
            basepath = Path(self.config.get("basepath", "."))
            if self.config.get("recursive", True):
                for path in basepath.rglob("*.py"):
                    # Check if path matches any ignore patterns
                    ignore = False
                    for pattern in self.config.get("ignore", []):
                        if path.match(pattern):
                            ignore = True
                            break
                    if not ignore:
                        files.append(str(path))
                    
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = []
            for file in files:
                if len(futures) >= self.config.get("throttle", 1000):
                    break
                futures.append(executor.submit(self._check_file, file, use_cache))
                
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                    
        # Show results
        if show_summary:
            self._show_summary(results)
            
        if show_details:
            self._show_details(results)
            
        if save_results:
            self._save_results(results, save_results)
            
        return len([r for r in results if r["status"] == "fail"]) == 0
        
    def _check_file(self, file: str, use_cache: bool) -> Optional[Dict]:
        """Check a single file.
        
        Args:
            file: Path to file
            use_cache: Whether to use cached results
            
        Returns:
            Check results or None if checks pass
        """
        import pylint.lint
        import mypy.api
        
        results = {
            "file": file,
            "status": "pass",
            "errors": []
        }
        
        # Run pylint
        try:
            pylint.lint.Run([file], do_exit=False)
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"Pylint error: {e}")
            
        # Run mypy
        mypy_args = [file]
        if use_cache:
            mypy_args.append("--cache-dir")
            mypy_args.append(str(self.workspace_root / ".mypy_cache"))
            
        try:
            mypy_result = mypy.api.run(mypy_args)
            if mypy_result[0]:  # stdout
                results["status"] = "fail"
                results["errors"].extend(mypy_result[0].splitlines())
        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(f"Mypy error: {e}")
            
        return results if results["status"] == "fail" else None
        
    def _show_summary(self, results: List[Dict]) -> None:
        """Show summary of check results.
        
        Args:
            results: List of check results
        """
        total = len(results)
        failed = len([r for r in results if r["status"] == "fail"])
        
        table = Table(title="Code Check Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Files", str(total))
        table.add_row("Failed Checks", str(failed))
        table.add_row("Pass Rate", f"{((total-failed)/total)*100:.1f}%")
        
        console.print(table)
        
    def _show_details(self, results: List[Dict]) -> None:
        """Show detailed check results.
        
        Args:
            results: List of check results
        """
        for result in results:
            console.print(f"\n[red]Errors in {result['file']}:[/red]")
            for error in result["errors"]:
                console.print(f"  {error}")
                
    def _save_results(self, results: List[Dict], filename: str) -> None:
        """Save check results to file.
        
        Args:
            results: List of check results
            filename: Output filename
        """
        with open(filename, "w") as f:
            toml.dump(results, f)
            
def main():
    """Run code checks from command line."""
    try:
        checker = CodeChecker()
        success = checker.run_checks()
        sys.exit(0 if success else 1)
    except Exception as e:
        console.print(f"[red]Error running code checks: {e}[/red]")
        sys.exit(1)
        
if __name__ == "__main__":
    main() 