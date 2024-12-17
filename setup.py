#!/usr/bin/env python3
"""Setup script for LlamaHome."""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import click
import torch
from rich.console import Console

from src.core.utils import (
    LogManager,
    LogTemplates,
    system_check,
    cache_manager
)
from src.core.model import ModelSetup

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)
console = Console()

def setup_environment() -> Dict[str, bool]:
    """Set up the environment.
    
    Returns:
        Dict containing setup results
    """
    results = {}
    
    # Check system requirements
    results['system_checks'] = system_check.check_system_requirements()
    
    # Create required directories
    required_dirs = [
        'data',
        'models',
        'config',
        'logs',
        'cache'
    ]
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(parents=True)
            results[f'created_{dir_name}'] = True
    
    # Initialize cache
    cache = cache_manager.CacheManager()
    results['cache_initialized'] = True
    
    return results

@click.command()
@click.option('--model', help='Model to set up')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def main(model: Optional[str], config: Optional[str]):
    """Set up LlamaHome environment."""
    try:
        # Set up environment
        results = setup_environment()
        if not all(results.values()):
            failed = [k for k, v in results.items() if not v]
            console.print(f"[red]Setup failed for: {', '.join(failed)}[/red]")
            sys.exit(1)
        
        # Set up model if specified
        if model:
            setup = ModelSetup(config_path=config)
            if not setup.setup_model(model):
                console.print("[red]Model setup failed[/red]")
                sys.exit(1)
        
        console.print("[green]Setup completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Setup failed: {e}[/red]")
        logger.exception("Setup failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
