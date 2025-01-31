"""Model creation and management functionality."""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import AutoModel, AutoConfig, PreTrainedModel

logger = logging.getLogger(__name__)


async def create_model(model_config: Dict[str, Any]) -> PreTrainedModel:
    """Create and initialize model from configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Initialized model
        
    Raises:
        ValueError: If model configuration is invalid
    """
    try:
        # Get model configuration
        model_name = model_config.get("name")
        if not model_name:
            raise ValueError("Model name not specified in configuration")

        # Load model configuration
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_config.get("config", {})
        )

        # Create model
        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        )

        # Apply any model-specific configurations
        if model_config.get("use_gradient_checkpointing", False):
            model.gradient_checkpointing_enable()

        if model_config.get("tie_word_embeddings", True):
            if hasattr(model, "tie_weights"):
                model.tie_weights()

        # Initialize weights if specified
        if model_config.get("init_weights", True):
            model.init_weights()

        return model

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise ValueError(f"Model creation failed: {e}")


def save_model(
    model: PreTrainedModel,
    path: str,
    tokenizer=None,
    optimizer=None,
    scheduler=None,
    training_args=None
) -> None:
    """Save model and related training components.
    
    Args:
        model: Model to save
        path: Path to save to
        tokenizer: Optional tokenizer to save
        optimizer: Optional optimizer state to save
        scheduler: Optional scheduler state to save
        training_args: Optional training arguments to save
    """
    try:
        # Save model
        model.save_pretrained(path)

        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

        # Save training state if provided
        if any([optimizer, scheduler, training_args]):
            training_state = {
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "training_args": training_args,
            }
            torch.save(training_state, f"{path}/training_state.pt")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def load_model(
    path: str,
    device: Optional[torch.device] = None,
    training_mode: bool = False
) -> PreTrainedModel:
    """Load saved model.
    
    Args:
        path: Path to load from
        device: Optional device to load model to
        training_mode: Whether to load model in training mode
        
    Returns:
        Loaded model
    """
    try:
        # Load configuration
        config = AutoConfig.from_pretrained(path)

        # Load model
        model = AutoModel.from_pretrained(
            path,
            config=config,
            trust_remote_code=True,
        )

        # Move to device if specified
        if device is not None:
            model = model.to(device)

        # Set training mode
        model.train(training_mode)

        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


class ModelManager:
    """Manages model lifecycle and operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model manager.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def initialize_model(self) -> PreTrainedModel:
        """Initialize model from configuration.
        
        Returns:
            Initialized model
        """
        self.model = await create_model(self.config)
        self.model.to(self.device)
        return self.model

    def save_checkpoint(
        self,
        path: str,
        optimizer=None,
        scheduler=None,
        training_args=None
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save to
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            training_args: Optional training arguments
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        save_model(
            self.model,
            path,
            optimizer=optimizer,
            scheduler=scheduler,
            training_args=training_args
        )

    def load_checkpoint(
        self,
        path: str,
        training_mode: bool = False
    ) -> PreTrainedModel:
        """Load model checkpoint.
        
        Args:
            path: Path to load from
            training_mode: Whether to load in training mode
            
        Returns:
            Loaded model
        """
        self.model = load_model(
            path,
            device=self.device,
            training_mode=training_mode
        )
        return self.model

    def prepare_for_training(self) -> None:
        """Prepare model for training."""
        if self.model is None:
            raise ValueError("Model not initialized")

        if self.config.get("use_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        if self.config.get("tie_word_embeddings", True):
            if hasattr(self.model, "tie_weights"):
                self.model.tie_weights()

    def prepare_for_inference(self) -> None:
        """Prepare model for inference."""
        if self.model is None:
            raise ValueError("Model not initialized")

        self.model.eval()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()


class ModelError(Exception):
    """Model-related error."""
    pass 