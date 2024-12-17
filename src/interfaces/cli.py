"""
CLI interface for LlamaHome.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import subprocess
import sys
import torch
import click
import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..training import (
    CacheConfig,
    DataConfig,
    MonitorConfig,
    OptimizationConfig,
    ProcessingConfig,
    ResourceConfig,
    TrainingConfig,
    TrainingPipeline
)

console = Console()

# Command groups
@click.group()
def cli():
    """LlamaHome CLI interface."""
    pass

# Training commands
@cli.group()
def train():
    """Training commands."""
    pass

@train.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--model', '-m', help='Model name/path', required=True)
@click.option('--output', '-o', help='Output directory', default='output/training')
@click.option('--config', '-c', help='Training config file', default=None)
@click.option('--eval-data', '-e', help='Evaluation data path', default=None)
async def start(data_path: str, model: str, output: str, config: Optional[str], eval_data: Optional[str]):
    """Start training a model."""
    try:
        # Load config
        training_config = _load_training_config(config)
        training_config.output_dir = output
        
        # Load model and tokenizer
        console.print("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            padding_side="left"
        )
        
        # Initialize pipeline
        console.print("Initializing training pipeline...")
        pipeline = TrainingPipeline(model, tokenizer, training_config)
        
        # Start training
        console.print("Starting training...")
        await pipeline.train(data_path, eval_data)
        console.print("[green]Training completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise click.Abort()

@train.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('--data-path', '-d', help='Training data path', required=True)
@click.option('--eval-data', '-e', help='Evaluation data path', default=None)
async def resume(checkpoint_path: str, data_path: str, eval_data: Optional[str]):
    """Resume training from a checkpoint."""
    try:
        # Load checkpoint
        console.print("Loading checkpoint...")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # Load config
        config_path = Path(checkpoint_path) / "training_config.yaml"
        training_config = _load_training_config(str(config_path))
        
        # Initialize pipeline
        console.print("Initializing training pipeline...")
        pipeline = TrainingPipeline(model, tokenizer, training_config)
        
        # Resume training
        console.print("Resuming training...")
        await pipeline.train(data_path, eval_data)
        console.print("[green]Training completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise click.Abort()

@train.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('eval_data', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory', default='output/eval')
async def evaluate(model_path: str, eval_data: str, output: str):
    """Evaluate a trained model."""
    try:
        # Load model
        console.print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config
        config_path = Path(model_path) / "training_config.yaml"
        training_config = _load_training_config(str(config_path))
        training_config.output_dir = output
        
        # Initialize pipeline
        console.print("Initializing evaluation pipeline...")
        pipeline = TrainingPipeline(model, tokenizer, training_config)
        
        # Run evaluation
        console.print("Starting evaluation...")
        metrics = await pipeline._evaluate(eval_data)
        
        # Save results
        results_path = Path(output) / "eval_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        console.print("[green]Evaluation completed successfully![/green]")
        console.print("Results saved to:", results_path)
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise click.Abort()

def _load_training_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load training configuration."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        # Load default config
        config_path = Path(".config/training_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = {}
    
    # Create config objects
    return TrainingConfig(
        cache=CacheConfig(**config_dict.get("cache", {})),
        data=DataConfig(**config_dict.get("data", {})),
        monitor=MonitorConfig(**config_dict.get("monitor", {})),
        optimization=OptimizationConfig(**config_dict.get("optimization", {})),
        processing=ProcessingConfig(**config_dict.get("processing", {})),
        resource=ResourceConfig(**config_dict.get("resource", {})),
        **config_dict.get("training", {})
    )

# Shell interface
@cli.command()
def shell():
    """Start interactive shell."""
    # Command completion
    commands = WordCompleter([
        'train', 'train-resume', 'train-eval',
        'help', 'exit', 'quit'
    ])
    
    # Create session
    session = PromptSession(completer=commands)
    
    async def run_shell():
        while True:
            try:
                # Get command
                text = await session.prompt_async('llamahome> ')
                command = text.strip()
                
                if command in ['exit', 'quit']:
                    console.print("Goodbye!")
                    break
                
                elif command == 'help':
                    console.print("Available commands:")
                    console.print("  train <data_path> - Start training")
                    console.print("  train-resume <checkpoint> - Resume training")
                    console.print("  train-eval <model> <data> - Evaluate model")
                    console.print("  help - Show this help message")
                    console.print("  exit/quit - Exit shell")
                
                elif command.startswith('train '):
                    args = command.split()[1:]
                    if len(args) < 1:
                        console.print("[red]Error: Missing data path[/red]")
                        continue
                    await start(args[0], args[1] if len(args) > 1 else None)
                
                elif command.startswith('train-resume '):
                    args = command.split()[1:]
                    if len(args) < 2:
                        console.print("[red]Error: Missing checkpoint or data path[/red]")
                        continue
                    await resume(args[0], args[1], args[2] if len(args) > 2 else None)
                
                elif command.startswith('train-eval '):
                    args = command.split()[1:]
                    if len(args) < 2:
                        console.print("[red]Error: Missing model or data path[/red]")
                        continue
                    await evaluate(args[0], args[1], args[2] if len(args) > 2 else 'output/eval')
                
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    # Run shell
    asyncio.run(run_shell())

@cli.group()
def test():
    """Testing commands."""
    pass

@test.command()
@click.option('--unit', is_flag=True, help='Run unit tests')
@click.option('--integration', is_flag=True, help='Run integration tests')
@click.option('--performance', is_flag=True, help='Run performance tests')
@click.option('--specialized', is_flag=True, help='Run specialized tests')
@click.option('--distributed', is_flag=True, help='Run distributed tests')
@click.option('--coverage', is_flag=True, help='Generate coverage report')
@click.option('--gpu', is_flag=True, help='Run GPU tests')
@click.option('--all', is_flag=True, help='Run all tests')
def run(unit, integration, performance, specialized, distributed, coverage, gpu, all):
    """Run tests."""
    try:
        args = []
        
        if all:
            args.extend(['tests/', '-v'])
        else:
            if unit:
                args.extend(['tests/', '-v', '-m', 'not integration and not performance and not specialized'])
            if integration:
                args.extend(['tests/', '-v', '-m', 'integration'])
            if performance:
                args.extend(['tests/performance/', '-v', '-m', 'performance'])
            if specialized:
                args.extend(['tests/specialized/', '-v', '-m', 'specialized'])
            if distributed:
                args.extend(['tests/distributed/', '-v', '-m', 'distributed'])
                
        if gpu:
            args.append('--gpu')
            
        if coverage:
            args.extend(['--cov=src', '--cov-report=html', '--cov-report=xml'])
            
        if not args:
            args.extend(['tests/', '-v', '-m', 'not integration and not performance and not specialized'])
            
        result = subprocess.run(['pytest'] + args, check=True)
        if result.returncode != 0:
            console.print("[red]Tests failed[/red]")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running tests: {e}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    cli()
