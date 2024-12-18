#!/usr/bin/env python3
"""Script to automatically fix common linting issues."""

import os
import sys
from pathlib import Path

def ensure_newline_at_eof(file_path: Path) -> None:
    """Ensure file ends with exactly one newline."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.rstrip() + '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def fix_blank_lines(file_path: Path) -> None:
    """Fix blank line issues (whitespace and count between functions/classes)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove trailing whitespace
    lines = [line.rstrip() + '\n' for line in lines]
    
    # Fix blank lines between functions/classes
    new_lines = []
    in_import_block = False
    prev_def = False
    consecutive_blanks = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Handle import blocks
        if stripped.startswith(('import ', 'from ')):
            in_import_block = True
        elif stripped and not stripped.startswith(('import ', 'from ')):
            if in_import_block:
                in_import_block = False
                if not stripped.startswith('#'):
                    new_lines.append('\n')
        
        # Handle function/class definitions
        if stripped.startswith(('def ', 'class ')):
            # Ensure two blank lines before definitions (except at file start)
            if new_lines and not prev_def:
                while len(new_lines) > 0 and new_lines[-1].strip() == '':
                    new_lines.pop()
                new_lines.extend(['\n', '\n'])
            elif prev_def:
                while len(new_lines) > 0 and new_lines[-1].strip() == '':
                    new_lines.pop()
                new_lines.extend(['\n', '\n'])
            prev_def = True
        elif stripped:
            prev_def = False
        
        # Handle consecutive blank lines
        if not stripped:
            consecutive_blanks += 1
            if consecutive_blanks <= 2:  # Allow at most 2 consecutive blank lines
                new_lines.append(line)
        else:
            consecutive_blanks = 0
            new_lines.append(line)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    """Main function to fix linting issues."""
    src_dir = Path('src')
    if not src_dir.exists():
        print("Error: src directory not found")
        sys.exit(1)
    
    python_files = list(src_dir.rglob('*.py'))
    for file_path in python_files:
        print(f"Processing {file_path}...")
        ensure_newline_at_eof(file_path)
        fix_blank_lines(file_path)

if __name__ == '__main__':
    main() 