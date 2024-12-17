"""Command Line Interface for LlamaHome."""

import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle, prompt
import os

from llama_stack import LlamaStack
from llama_stack_client import LlamaClient

from ..core.model_manager import ModelManager
from ..core.setup import ModelSetup

console = Console()

class CLIInterface:
    """Command Line Interface handler."""
    
    def __init__(self):
        """Initialize CLI interface."""
        self.llama = LlamaStack()
        self.client = LlamaClient()
        self.model_manager = ModelManager()
        self.model_setup = ModelSetup()
        self.running = True
        
        # Set up command history
        history_file = Path(".config/history.txt")
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(history_file))
        
        # Set up command completion
        self.commands = [
            'help', 'models', 'chat', 'quit', 'exit',
            'llama', 'llama-install', 'llama-update'
        ]
        self.completer = WordCompleter(self.commands, ignore_case=True)
        
        # Set up key bindings
        self.bindings = KeyBindings()
        
        @self.bindings.add('c-c')
        def _(event):
            """Handle Ctrl+C."""
            self.quit()
            event.app.exit()
        
        # Create prompt session
        self.session = PromptSession(
            history=self.history,
            completer=self.completer,
            auto_suggest=AutoSuggestFromHistory(),
            complete_style=CompleteStyle.MULTI_COLUMN,
            key_bindings=self.bindings,
            enable_history_search=True,
            mouse_support=True,
            complete_while_typing=True
        )

    def start(self):
        """Start the CLI interface."""
        console.print("\n[bold]Welcome to LlamaHome![/bold]\n")
        
        # Show help menu on startup
        self.show_help()
        
        while self.running:
            try:
                # Get command with autocompletion and history
                command = self.session.prompt(
                    "\nEnter command: ",
                    default='help'
                ).strip().lower()
                
                if command:
                    self.handle_command(command)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                self.quit()
                break

    def update_completions(self):
        """Update command completions based on available models."""
        # Get available models and versions
        available_models = self.model_manager.list_available_models()
        model_versions = []
        for model, versions in available_models.items():
            model_versions.extend([f"{model} {version}" for version in versions])
        
        # Update completions
        completions = self.commands + [
            'model ' + m for m in available_models.keys()
        ] + [
            'download ' + mv for mv in model_versions
        ] + [
            'remove ' + m for m in available_models.keys()
        ] + [
            'chat ' + m for m in available_models.keys()
        ] + [
            'train ' + m for m in available_models.keys()
        ]
        
        self.completer = WordCompleter(completions, ignore_case=True)
        self.session.completer = self.completer

    def handle_command(self, command: str):
        """Handle CLI commands."""
        if command == "help":
            self.show_help()
        elif command in ["quit", "exit"]:
            self.quit()
        elif command == "models":
            self.list_models()
        elif command.startswith("model "):
            self.show_model_details(command.split(" ")[1])
        elif command.startswith("download "):
            self.handle_download(command)
        elif command.startswith("remove "):
            self.handle_remove(command)
        elif command.startswith("chat"):
            self.handle_chat(command)
        elif command.startswith("train"):
            self.handle_train(command)
        elif command == "llama":
            self.show_llama_help()
        elif command == "llama-install":
            self.install_llama_stack()
        elif command == "llama-update":
            self.update_llama_stack()
        elif command.startswith("llama "):
            self.handle_llama_command(command)
        else:
            console.print("[red]Unknown command. Type 'help' for options.[/red]")

    def show_help(self):
        """Show help information."""
        console.print(Panel.fit(
            "Available commands:\n"
            "help - Show this help message\n"
            "models - List available models\n"
            "model <name> - Show model details\n"
            "download <model> <version> - Download a model\n"
            "remove <model> [version] - Remove a model\n"
            "chat [model] [version] - Start chat session\n"
            "train <model> [version] - Train model on local data\n"
            "llama - Show llama-stack commands\n"
            "llama-install - Install llama-stack\n"
            "llama-update - Update llama-stack\n"
            "quit/exit - Exit application",
            title="Commands"
        ))

    def list_models(self):
        """List available models."""
        available_models = self.model_manager.list_available_models()
        
        model_info = []
        for model_type, versions in available_models.items():
            config = self.model_manager.config.model_configs[model_type]
            model_info.append(f"{model_type}:")
            model_info.append(f"  Versions: {', '.join(versions)}")
            model_info.append(f"  Format: {', '.join(config['formats'])}")
            if 'api_required' in config:
                model_info.append("  API Required: Yes")
            if 'min_gpu_memory' in config:
                model_info.append("  GPU Memory Requirements:")
                for ver, mem in config['min_gpu_memory'].items():
                    model_info.append(f"    {ver}: {mem}GB")
            model_info.append("")
        
        console.print(Panel.fit(
            "\n".join(model_info),
            title="Available Models"
        ))

    def show_model_details(self, model_name: str):
        """Show details for a specific model.
        
        Args:
            model_name: Name of the model
        """
        if model_name in self.model_manager.config.model_types:
            config = self.model_manager.config.model_configs[model_name]
            versions = self.model_manager.list_available_models().get(model_name, [])
            
            model_info = [
                f"Model: {model_name}",
                f"Available Versions: {', '.join(versions)}",
                f"Default Version: {config['default_version']}",
                f"Format: {', '.join(config['formats'])}",
            ]
            
            if 'api_required' in config:
                model_info.append("API Required: Yes")
            if 'min_gpu_memory' in config:
                model_info.append("GPU Memory Requirements:")
                for ver, mem in config['min_gpu_memory'].items():
                    model_info.append(f"  {ver}: {mem}GB")
            
            console.print(Panel.fit(
                "\n".join(model_info),
                title=f"Model Details: {model_name}"
            ))
        else:
            console.print(f"[red]Unknown model: {model_name}[/red]")

    def handle_download(self, command: str):
        """Handle model download command.
        
        Args:
            command: The download command
        """
        parts = command.split(" ")
        if len(parts) != 3:
            console.print("[red]Usage: download <model> <version>[/red]")
            return
        
        model_name = parts[1]
        version = parts[2]
        
        if model_name not in self.model_manager.config.model_types:
            console.print(f"[red]Unknown model: {model_name}[/red]")
            return
        
        config = self.model_manager.config.model_configs[model_name]
        if version not in config["versions"]:
            console.print(f"[red]Invalid version {version} for {model_name}[/red]")
            return
        
        try:
            success = self.model_setup.setup_model(model_name, version, force_setup=True)
            if success:
                console.print(f"[green]Successfully downloaded {model_name} {version}[/green]")
            else:
                console.print(f"[red]Failed to download {model_name} {version}[/red]")
        except Exception as e:
            console.print(f"[red]Error downloading model: {e}[/red]")

    def handle_remove(self, command: str):
        """Handle model remove command.
        
        Args:
            command: The remove command
        """
        parts = command.split(" ")
        if len(parts) < 2:
            console.print("[red]Usage: remove <model> [version][/red]")
            return
        
        model_name = parts[1]
        if model_name not in self.model_manager.config.model_types:
            console.print(f"[red]Unknown model: {model_name}[/red]")
            return
        
        try:
            # If version is specified, remove that specific version
            if len(parts) > 2:
                version = parts[2]
                if version not in self.model_manager.config.model_configs[model_name]["versions"]:
                    console.print(f"[red]Invalid version {version} for {model_name}[/red]")
                    return
                
                model_path = self.model_manager.get_model_path(model_name, version)
                if not model_path.exists():
                    console.print(f"[yellow]Model {model_name} {version} not found[/yellow]")
                    return
                
                self.model_manager.cleanup_model_files(model_name, version)
                console.print(f"[green]Successfully removed {model_name} {version}[/green]")
            
            # If no version specified, remove all versions
            else:
                self.model_manager.cleanup_model_files(model_name)
                console.print(f"[green]Successfully removed all {model_name} models[/green]")
                
        except Exception as e:
            console.print(f"[red]Error removing model: {e}[/red]")

    def handle_chat(self, command: str):
        """Handle chat command.
        
        Args:
            command: The chat command
        """
        parts = command.split()
        
        # Use current model/version if set, otherwise use command arguments
        model_name = self.current_model
        version = self.current_version
        
        if len(parts) > 1:
            model_name = parts[1]
            if len(parts) > 2:
                version = parts[2]
        
        if not model_name:
            model_name = "llama"  # Default model
        
        if not version:
            version = self.model_manager.config.model_configs[model_name]["default_version"]
        
        # Validate model and version
        if model_name not in self.model_manager.config.model_types:
            console.print(f"[red]Unknown model: {model_name}[/red]")
            return
            
        if version not in self.model_manager.config.model_configs[model_name]["versions"]:
            console.print(f"[red]Invalid version {version} for {model_name}[/red]")
            return
            
        # Check if model is downloaded
        if not self.model_manager.validate_model_files(model_name, version):
            console.print(f"[red]Model {model_name} {version} not found. Please download it first.[/red]")
            return
            
        console.print(f"[green]Starting chat with {model_name} {version}...[/green]")
        self.current_model = model_name
        self.current_version = version
        
        # Load model if needed
        if not self.model or not self.tokenizer:
            console.print("[yellow]Loading model...[/yellow]")
            if not self._load_model(model_name, version):
                return
        
        # Start chat loop
        while True:
            user_input = prompt("\nYou")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break
            
            # Generate and display response
            console.print("\n[cyan]Assistant:[/cyan] ", end="")
            self._generate_response(user_input)

    def handle_train(self, command: str):
        """Handle train command.
        
        Args:
            command: The train command
        """
        parts = command.split()
        if len(parts) < 2:
            console.print("[red]Usage: train <model> [version][/red]")
            return
            
        model_name = parts[1]
        version = parts[2] if len(parts) > 2 else None
        
        if model_name not in self.model_manager.config.model_types:
            console.print(f"[red]Unknown model: {model_name}[/red]")
            return
            
        if version and version not in self.model_manager.config.model_configs[model_name]["versions"]:
            console.print(f"[red]Invalid version {version} for {model_name}[/red]")
            return
            
        try:
            console.print("[green]Processing training samples...[/green]")
            # Use asyncio.run since these are async methods
            import asyncio
            asyncio.run(self.training_manager.process_samples(model_name, version))
            
            console.print("[green]Starting training...[/green]")
            asyncio.run(self.training_manager.train_model(model_name, version))
            
            console.print("[green]Training completed successfully![/green]")
        except Exception as e:
            console.print(f"[red]Error during training: {e}[/red]")

    def quit(self):
        """Exit the application."""
        console.print("[green]Goodbye! Thank you for using LlamaHome.[/green]")
        self.running = False

    def show_llama_help(self):
        """Show llama-stack specific help."""
        console.print(Panel.fit(
            "Llama Stack Commands:\n"
            "llama model list - List all available models\n"
            "llama model list --show-all - Show all model versions\n"
            "llama model download <model-id> - Download a specific model\n"
            "\nExample Commands:\n"
            "llama model list\n"
            "llama model list --show-all\n"
            "llama model download Llama-3.3-70B-Instruct\n"
            "\nNote: Model downloads will use settings from your .env file:\n"
            "- LLAMA_MODEL_SIZE (default: 13b)\n"
            "- LLAMA_MODEL_VARIANT (default: chat)\n"
            "- LLAMA_MODEL_QUANT (default: f16)\n"
            "- LLAMA_NUM_GPU_LAYERS (default: 32)\n"
            "- LLAMA_MAX_SEQ_LEN (default: 32768)\n"
            "- LLAMA_MAX_BATCH_SIZE (default: 8)",
            title="Llama Stack Help"
        ))

    def install_llama_stack(self):
        """Install llama-stack package."""
        console.print("Installing llama-stack packages...")
        try:
            # Install both packages
            result1 = subprocess.run(
                ["pip", "install", "-U", "llama-stack"],
                capture_output=True,
                text=True
            )
            result2 = subprocess.run(
                ["pip", "install", "-U", "llama-stack-client"],
                capture_output=True,
                text=True
            )
            
            if result1.returncode == 0 and result2.returncode == 0:
                console.print("[green]Successfully installed llama-stack packages[/green]")
                # Verify installation by checking version
                self._verify_llama_stack()
            else:
                console.print("[red]Failed to install llama-stack packages:[/red]")
                if result1.returncode != 0:
                    console.print(f"[red]llama-stack: {result1.stderr}[/red]")
                if result2.returncode != 0:
                    console.print(f"[red]llama-stack-client: {result2.stderr}[/red]")
        except Exception as e:
            console.print(f"[red]Error installing llama-stack: {e}[/red]")

    def update_llama_stack(self):
        """Update llama-stack package."""
        console.print("Updating llama-stack packages...")
        try:
            # Update both packages
            result1 = subprocess.run(
                ["pip", "install", "-U", "llama-stack"],
                capture_output=True,
                text=True
            )
            result2 = subprocess.run(
                ["pip", "install", "-U", "llama-stack-client"],
                capture_output=True,
                text=True
            )
            
            if result1.returncode == 0 and result2.returncode == 0:
                console.print("[green]Successfully updated llama-stack packages[/green]")
                # Verify installation by checking version
                self._verify_llama_stack()
            else:
                console.print("[red]Failed to update llama-stack packages:[/red]")
                if result1.returncode != 0:
                    console.print(f"[red]llama-stack: {result1.stderr}[/red]")
                if result2.returncode != 0:
                    console.print(f"[red]llama-stack-client: {result2.stderr}[/red]")
        except Exception as e:
            console.print(f"[red]Error updating llama-stack: {e}[/red]")

    def _verify_llama_stack(self):
        """Verify llama-stack installation."""
        try:
            # Check both packages
            result1 = subprocess.run(
                ["python", "-c", "import llama_stack; print(llama_stack.__version__)"],
                capture_output=True,
                text=True
            )
            result2 = subprocess.run(
                ["python", "-c", "import llama_stack_client; print(llama_stack_client.__version__)"],
                capture_output=True,
                text=True
            )
            
            if result1.returncode == 0 and result2.returncode == 0:
                console.print(f"[green]llama-stack version: {result1.stdout.strip()}[/green]")
                console.print(f"[green]llama-stack-client version: {result2.stdout.strip()}[/green]")
                return True
            return False
        except Exception:
            return False

    def handle_llama_command(self, command: str):
        """Handle llama-stack specific commands."""
        # First verify llama-stack is installed
        if not self._verify_llama_stack():
            console.print("[red]llama-stack is not installed. Please run 'llama-install' first.[/red]")
            return

        parts = command.split()
        if len(parts) < 2:
            self.show_llama_help()
            return

        if parts[1] == "model":
            if len(parts) < 3:
                self.show_llama_help()
                return

            if parts[2] == "list":
                # Handle model listing
                try:
                    from llama_stack_client import LlamaClient
                    client = LlamaClient()
                    if len(parts) > 3 and parts[3] == "--show-all":
                        models = client.list_models(show_all=True)
                    else:
                        models = client.list_models()
                    
                    # Format and display models
                    console.print("\n[bold]Available Models:[/bold]\n")
                    for category, model_list in models.items():
                        console.print(f"[cyan]{category}:[/cyan]")
                        for model in model_list:
                            console.print(f"  - {model}")
                        console.print()
                        
                except Exception as e:
                    console.print(f"[red]Error listing models: {e}[/red]")

            elif parts[2] == "download" and len(parts) > 3:
                # Handle model download
                model_id = parts[3]
                try:
                    from llama_stack_client import LlamaClient
                    client = LlamaClient()
                    console.print(f"Downloading model {model_id}...")
                    
                    # Add download options from environment variables
                    options = {
                        "model_size": os.getenv("LLAMA_MODEL_SIZE", "13b"),
                        "model_variant": os.getenv("LLAMA_MODEL_VARIANT", "chat"),
                        "model_quant": os.getenv("LLAMA_MODEL_QUANT", "f16"),
                        "num_gpu_layers": int(os.getenv("LLAMA_NUM_GPU_LAYERS", "32")),
                        "max_seq_len": int(os.getenv("LLAMA_MAX_SEQ_LEN", "32768")),
                        "max_batch_size": int(os.getenv("LLAMA_MAX_BATCH_SIZE", "8"))
                    }
                    
                    client.download_model(
                        model_id,
                        source="meta",
                        output_dir=str(self.model_manager.get_model_path("llama", model_id)),
                        **options
                    )
                    console.print(f"[green]Successfully downloaded model {model_id}[/green]")
                except Exception as e:
                    console.print(f"[red]Error downloading model: {e}[/red]")
            else:
                self.show_llama_help()
        else:
            self.show_llama_help()
