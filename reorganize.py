"""Script to reorganize LlamaHome project structure."""

import os
import shutil
from pathlib import Path

def safe_move(src: Path, dst: Path) -> None:
    """Safely move a file or directory."""
    if src.is_file():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(str(src), str(dst))
    elif src.is_dir():
        if not dst.exists():
            shutil.move(str(src), str(dst))
        else:
            # Merge directories
            for item in src.glob('*'):
                target = dst / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))

def reorganize_project():
    """Reorganize project structure."""
    # Create new directory structure
    new_dirs = [
        "src/core/models",
        "src/core/attention",
        "src/core/config",
        "src/data/processing",
        "src/data/storage",
        "src/data/validation",
        "tests/unit",
        "tests/integration",
        "tests/performance/benchmarks",
        "tests/fixtures",
        "tools",
        "data/cache"
    ]
    
    for dir_path in new_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Move files to new locations
    moves = [
        # Core reorganization
        ("src/core/model.py", "src/core/models/base.py"),
        ("src/core/attention.py", "src/core/attention/hybrid.py"),
        ("src/core/model_constants.py", "src/core/config/constants.py"),
        
        # Data reorganization
        ("src/data/analyzer.py", "src/data/processing/analyzer.py"),
        ("src/data/converter.py", "src/data/processing/converter.py"),
        ("src/data/storage.py", "src/data/storage/base.py"),
        
        # Utils consolidation
        ("utils/benchmark.py", "src/utils/benchmark.py"),
        ("utils/code_check.py", "tools/code_check.py"),
        ("utils/system_check.py", "tools/system_check.py"),
        ("utils/yaml_to_toml_converter.py", "tools/yaml_to_toml_converter.py"),
        
        # Test reorganization
        ("tests/core", "tests/unit/core"),
        ("tests/data", "tests/unit/data"),
        ("tests/interfaces", "tests/unit/interfaces"),
        ("tests/specialized", "tests/integration/specialized")
    ]
    
    # Move performance tests
    perf_dir = Path("tests/performance")
    if perf_dir.exists():
        for item in perf_dir.glob('*'):
            if item.is_file():
                shutil.move(str(item), str(Path("tests/performance/benchmarks") / item.name))
    
    # Move files
    for src, dst in moves:
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.exists():
            safe_move(src_path, dst_path)
    
    # Remove .bak files
    for bak_file in Path(".").rglob("*.bak"):
        bak_file.unlink()
    
    # Consolidate managers
    manager_files = {
        "model_manager.py": "src/core/models/manager.py",
        "config_manager.py": "src/core/config/manager.py",
        "cache_manager.py": "src/data/storage/cache_manager.py",
        "data_manager.py": "src/data/storage/data_manager.py",
        "log_manager.py": "src/utils/log_manager.py",
        "training_manager.py": "src/training/manager.py"
    }
    
    for src_name, dst_path in manager_files.items():
        src_files = list(Path(".").rglob(src_name))
        if src_files:
            # Use the most recently modified file
            newest_file = max(src_files, key=lambda p: p.stat().st_mtime)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(str(newest_file), dst_path)
            # Remove all copies
            for f in src_files:
                f.unlink()

if __name__ == "__main__":
    reorganize_project() 