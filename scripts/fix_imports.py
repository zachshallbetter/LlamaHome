#!/usr/bin/env python3
"""Script to clean up unused imports."""

import ast
import sys
from pathlib import Path
from typing import Set, List


def get_used_names(tree: ast.AST) -> Set[str]:
    """Get all used names in the AST."""
    used_names = set()
    
    class NameVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            used_names.add(node.id)
            self.generic_visit(node)
        
        def visit_Attribute(self, node: ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
            self.generic_visit(node)
    
    NameVisitor().visit(tree)
    return used_names

def get_imported_names(tree: ast.AST) -> Set[str]:
    """Get all imported names in the AST."""
    imported_names = set()
    
    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        
        def visit_ImportFrom(self, node: ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    
    ImportVisitor().visit(tree)
    return imported_names

def fix_imports(file_path: Path) -> None:
    """Fix unused imports in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Syntax error in {file_path}")
        return
    
    used_names = get_used_names(tree)
    imported_names = get_imported_names(tree)
    unused_imports = imported_names - used_names
    
    if not unused_imports:
        return
    
    print(f"\nFixing imports in {file_path}")
    print(f"Unused imports: {', '.join(sorted(unused_imports))}")
    
    # Remove unused imports
    lines = content.split('\n')
    new_lines: List[str] = []
    
    for line in lines:
        # Skip lines that import unused names
        if any(f"import {name}" in line or f"as {name}" in line for name in unused_imports):
            continue
        new_lines.append(line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

def main():
    """Main function."""
    src_dir = Path('src')
    if not src_dir.exists():
        print("Error: src directory not found")
        sys.exit(1)
    
    python_files = list(src_dir.rglob('*.py'))
    for file_path in python_files:
        fix_imports(file_path)

if __name__ == '__main__':
    main() 