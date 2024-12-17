#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import toml
import toml
import re
from typing import Dict, Any, List, Tuple
import shutil

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML file and return dictionary."""
    with open(file_path, 'r') as f:
        return toml.load(f)

def save_toml(data: Dict[str, Any], file_path: Path) -> None:
    """Save dictionary as TOML file."""
    with open(file_path, 'w') as f:
        toml.dump(data, f)

def convert_file(yaml_path: Path) -> Path:
    """Convert a YAML file to TOML format."""
    data = load_yaml(yaml_path)
    toml_path = yaml_path.with_suffix('.toml')
    save_toml(data, toml_path)
    return toml_path

def backup_file(file_path: Path) -> None:
    """Create a backup of the file with .bak extension."""
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
    shutil.copy2(file_path, backup_path)

def update_file_content(file_path: Path) -> None:
    """Update file content to replace YAML references with TOML."""
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()

    # Update imports
    content = re.sub(r'import toml', 'import toml', content)
    content = re.sub(r'from toml', 'from toml', content)
    
    # Update file extensions
    content = re.sub(r'\.ya?ml([^a-zA-Z])', r'.toml\1', content)
    
    # Update yaml function calls
    content = re.sub(r'yaml\.safe_load', 'toml.load', content)
    content = re.sub(r'yaml\.dump', 'toml.dump', content)
    content = re.sub(r'yaml\.YAMLError', 'toml.TomlDecodeError', content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def find_files_to_update(workspace_root: Path) -> Tuple[List[Path], List[Path]]:
    """Find all YAML config files and Python files that need updating."""
    yaml_files = []
    python_files = []
    doc_files = []
    
    for path in workspace_root.rglob('*'):
        if path.is_file():
            if path.suffix in ('.toml', '.toml') and '.config' in str(path):
                yaml_files.append(path)
            elif path.suffix == '.py':
                with open(path, 'r') as f:
                    content = f.read()
                    if 'yaml' in content or '.toml' in content or '.toml' in content:
                        python_files.append(path)
            elif path.suffix in ('.md', '.rst'):
                with open(path, 'r') as f:
                    content = f.read()
                    if '.toml' in content or '.toml' in content:
                        doc_files.append(path)
    
    return yaml_files, python_files + doc_files

def main():
    workspace_root = Path(__file__).parent.parent
    yaml_files, files_to_update = find_files_to_update(workspace_root)
    
    print(f"Found {len(yaml_files)} YAML files to convert")
    print(f"Found {len(files_to_update)} files to update references")
    
    # Convert YAML files to TOML
    for yaml_file in yaml_files:
        print(f"Converting {yaml_file}")
        toml_path = convert_file(yaml_file)
        print(f"Created {toml_path}")
        
    # Update references in Python and documentation files
    for file_path in files_to_update:
        print(f"Updating references in {file_path}")
        update_file_content(file_path)
    
    print("\nConversion complete!")
    print("Please review the changes and test thoroughly before committing.")
    print("Backup files have been created with .bak extension.")

if __name__ == '__main__':
    main() 