"""Training data management and preprocessing."""

import json
import toml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from src.core.models import BenchmarkManager, ModelManager
from src.core.utils import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


class ProgressCallback(TrainerCallback):
    """Custom callback for training progress."""


    def __init__(self):
        """Initialize progress callback."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.task = None
        self.epoch_task = None


    def on_train_begin(self, args, state, control, **kwargs):
        """Handle training start."""
        self.progress.start()
        self.task = self.progress.add_task(
            "Training",
            total=args.num_train_epochs * state.max_steps
        )
        self.epoch_task = self.progress.add_task(
            "Current epoch",
            total=state.max_steps
        )


    def on_step_end(self, args, state, control, **kwargs):
        """Handle step end."""
        self.progress.update(self.task, advance=1)
        self.progress.update(self.epoch_task, advance=1)


    def on_epoch_begin(self, args, state, control, **kwargs):
        """Handle epoch start."""
        self.progress.reset(self.epoch_task)


    def on_train_end(self, args, state, control, **kwargs):
        """Handle training end."""
        self.progress.stop()


class ConversationDataset(Dataset):
    """Dataset for conversation samples."""


    def __init__(self, conversations: List[Dict], tokenizer, max_length: int = 512):
        """Initialize dataset.

        Args:
            conversations: List of conversation dictionaries
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.conversations)


    def __getitem__(self, idx):
        conv = self.conversations[idx]
        # Format conversation into a single string with special tokens
        text = ""
        for msg in conv["conversation"]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                text += f"<|user|>{content}<|enduser|>"
            else:
                text += f"<|assistant|>{content}<|endassistant|>"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For causal language modeling
        }


class TrainingData:
    """Manages training data processing and storage."""


    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 4,
        max_workers: int = 4,
        config: Optional[Dict] = None
    ) -> None:
        """Initialize training data manager.

        Args:
            data_dir: Directory for training data
            batch_size: Size of training batches
            max_workers: Maximum number of worker processes
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.config = self._load_config(config)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = BenchmarkManager()
        self.model_manager = ModelManager()
        logger.info(f"Initialized training manager with data dir: {self.data_dir}")


    def _load_config(self, config: Optional[Dict] = None) -> Dict:
        """Load training configuration.

        Args:
            config: Optional configuration override

        Returns:
            Loaded configuration
        """
        config_path = Path(".config/training_config.toml")
        if config_path.exists():
            with open(config_path) as f:
                base_config = toml.load(f)["training"]
        else:
            base_config = {}

        if config:
            # Deep merge configurations
            for key, value in config.items():
                if isinstance(value, dict) and key in base_config:
                    base_config[key].update(value)
                else:
                    base_config[key] = value

        return base_config


    def _get_model_config(self, model_name: str) -> Dict:
        """Get model-specific configuration.

        Args:
            model_name: Name of model

        Returns:
            Model configuration
        """
        config = self.config.copy()
        if "model_configs" in config and model_name in config["model_configs"]:
            model_config = config["model_configs"][model_name]
            # Deep merge configurations
            for key, value in model_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
        return config

    async def process_samples(
        self,
        model_name: str = "llama",
        model_version: Optional[str] = None
    ) -> None:
        """Process and prepare training samples.

        Args:
            model_name: Name of model to train
            model_version: Optional specific model version
        """
        # Load samples from JSONL files
        samples = []
        sample_files = list(self.data_dir.glob("samples/*.jsonl"))

        if not sample_files:
            logger.warning(f"No sample files found in {self.data_dir}/samples/")
            return

        logger.info(f"Found {len(sample_files)} sample files")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        ) as progress:
            load_task = progress.add_task("Loading samples...", total=len(sample_files))

            for file in sample_files:
                with open(file) as f:
                    for line in f:
                        if line.strip():
                            try:
                                sample = json.loads(line)
                                samples.append(sample)
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing sample in {file}: {e}")
                progress.update(load_task, advance=1)

        if not samples:
            logger.warning("No valid samples found")
            return

        logger.info(f"Loaded {len(samples)} samples")

        # Get model tokenizer
        model_path = self.model_manager.get_model_path(model_name, model_version)
        if not model_path.exists():
            logger.error(f"Model {model_name} {model_version} not found")
            return

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Add special tokens for conversation format
        special_tokens = {
            "additional_special_tokens": [
                "<|user|>", "<|enduser|>",
                "<|assistant|>", "<|endassistant|>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)

        # Get model-specific config
        model_config = self._get_model_config(model_name)

        # Create dataset
        dataset = ConversationDataset(
            samples,
            tokenizer,
            max_length=model_config.get("max_length", 512)
        )

        # Split into train/validation
        val_split = model_config.get("validation_split", 0.1)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.get("batch_size", 4),
            shuffle=True,
            num_workers=self.max_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config.get("batch_size", 4),
            shuffle=False,
            num_workers=self.max_workers
        )

        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Save processed data
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "train_samples": [train_dataset[i] for i in range(len(train_dataset))],
                "val_samples": [val_dataset[i] for i in range(len(val_dataset))],
                "config": {
                    "model_name": model_name,
                    "model_version": model_version,
                    "special_tokens": special_tokens,
                    **model_config
                }
            },
            processed_dir / f"{model_name}_{model_version}_data.pt"
        )

        logger.info("Completed processing samples")

    async def train_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None
    ) -> None:
        """Train model on processed data using LoRA fine-tuning.

        Args:
            model_name: Name of model to train
            model_version: Optional specific model version
            num_epochs: Optional number of training epochs
            learning_rate: Optional learning rate for training
            lora_r: Optional LoRA attention dimension
            lora_alpha: Optional LoRA alpha parameter
            lora_dropout: Optional LoRA dropout value
        """
        # Load processed data
        processed_file = self.data_dir / "processed" / f"{model_name}_{model_version}_data.pt"
        if not processed_file.exists():
            logger.error(f"No processed data found for {model_name} {model_version}")
            return

        data = torch.load(processed_file)
        train_samples = data["train_samples"]
        val_samples = data["val_samples"]
        config = data["config"]

        # Get model-specific config
        model_config = self._get_model_config(model_name)

        # Override with provided parameters
        if num_epochs is not None:
            model_config["epochs"] = num_epochs
        if learning_rate is not None:
            model_config["learning_rate"] = learning_rate
        if lora_r is not None:
            model_config["lora"]["r"] = lora_r
        if lora_alpha is not None:
            model_config["lora"]["alpha"] = lora_alpha
        if lora_dropout is not None:
            model_config["lora"]["dropout"] = lora_dropout

        # Initialize model and tokenizer
        model_path = self.model_manager.get_model_path(model_name, model_version)
        if not model_path.exists():
            logger.error(f"Model {model_name} {model_version} not found")
            return

        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Add special tokens
        if "special_tokens" in config:
            tokenizer.add_special_tokens(config["special_tokens"])
            model.resize_token_embeddings(len(tokenizer))

        # Prepare model for k-bit training if using quantization
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=model_config["lora"]["r"],
            lora_alpha=model_config["lora"]["alpha"],
            target_modules=model_config["lora"]["target_modules"],
            lora_dropout=model_config["lora"]["dropout"],
            bias=model_config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)

        # Create datasets
        train_dataset = ConversationDataset(
            train_samples,
            tokenizer,
            max_length=model_config.get("max_length", 512)
        )

        val_dataset = ConversationDataset(
            val_samples,
            tokenizer,
            max_length=model_config.get("max_length", 512)
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.data_dir / "checkpoints" / f"{model_name}_{model_version}"),
            num_train_epochs=model_config["epochs"],
            per_device_train_batch_size=model_config["batch_size"],
            per_device_eval_batch_size=model_config["batch_size"],
            gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
            warmup_steps=model_config["warmup_steps"],
            fp16=model_config["fp16"],
            logging_steps=model_config["logging"]["steps"],
            evaluation_strategy="steps",
            eval_steps=model_config["logging"]["eval_steps"],
            save_strategy=model_config["checkpointing"]["save_strategy"],
            save_total_limit=model_config["checkpointing"]["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        # Initialize trainer with callbacks
        callbacks = [ProgressCallback()]
        if model_config["early_stopping"]["enabled"]:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=model_config["early_stopping"]["patience"],
                    early_stopping_threshold=model_config["early_stopping"]["min_delta"]
                )
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=callbacks
        )

        logger.info(f"Starting training for {model_name} {model_version}")
        logger.info(f"Training parameters: {model_config}")

        # Train the model
        trainer.train()

        # Save the trained model
        output_dir = self.data_dir / "models" / f"{model_name}_{model_version}_finetuned"
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Save training metrics
        metrics = trainer.state.log_history
        metrics_file = output_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Model and metrics saved to {output_dir}")

    async def load_data(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.max_workers
        ) if self.train_dataset else None

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.max_workers
        ) if self.val_dataset else None

        return train_loader, val_loader


def create_training(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    max_workers: int = 4,
    config: Optional[Dict] = None
) -> TrainingData:
    """Create a new TrainingData instance.

    Args:
        data_dir: Directory for training data
        batch_size: Size of training batches
        max_workers: Maximum number of worker processes
        config: Optional configuration dictionary

    Returns:
        Configured TrainingData instance
    """
    return TrainingData(data_dir, batch_size, max_workers, config)
