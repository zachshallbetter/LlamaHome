"""Enhanced Llama model implementation with hybrid optimizations."""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import FloatTensor, LongTensor, Tensor
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from llama_recipes.utils.dataset import ConcatDataset
from llama_recipes.configs import train_config
from llama_recipes.utils.train_utils import (
    train,
    save_model,
    get_dataloader,
)

from .attention import HybridAttention
from .cache import Cache, DynamicCache
from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass
class ModelConfig:
    """Enhanced model configuration."""
    
    # Attention settings
    use_flash_attention: bool = True
    use_memory_efficient: bool = True
    sliding_window: Optional[int] = None
    
    # Cache settings
    initial_cache_size: int = 2048
    max_cache_size: Optional[int] = None
    
    # Training settings
    lora_config: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None
    
    # Performance settings
    use_cuda_graphs: bool = True
    use_kernel_opt: bool = True
    
    def __post_init__(self):
        """Set default configurations if needed."""
        if self.lora_config is None:
            self.lora_config = {
                "r": 8,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            }
        
        if self.quantization_config is None:
            self.quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16"
            }


class EnhancedLlamaForCausalLM(LlamaForCausalLM):
    """Enhanced Llama model with hybrid optimizations."""

    def __init__(self, config: LlamaConfig) -> None:
        """Initialize enhanced model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Set up model configuration
        self.model_config = ModelConfig(
            use_flash_attention=getattr(config, "use_flash_attention", True),
            use_memory_efficient=getattr(config, "use_memory_efficient", True),
            sliding_window=getattr(config, "sliding_window", None),
            initial_cache_size=getattr(config, "initial_cache_size", 2048),
            max_cache_size=getattr(config, "max_cache_size", None),
            use_cuda_graphs=getattr(config, "use_cuda_graphs", True),
            use_kernel_opt=getattr(config, "use_kernel_opt", True)
        )
        
        # Replace standard attention with hybrid attention
        num_layers = len(self.model.layers)
        logger.debug(f"Initializing enhanced model with {num_layers} layers")
        
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = HybridAttention(config, layer_idx)
            logger.debug(f"Initialized hybrid attention for layer {layer_idx}")

        # Configure model parameters
        self.model_name = "EnhancedLlama"
        self.cache_config = {
            "window_length": self.model_config.initial_cache_size,
            "max_length": self.model_config.max_cache_size
        }
        
        # Initialize CUDA graphs if enabled
        if self.model_config.use_cuda_graphs and torch.cuda.is_available():
            self._init_cuda_graphs()
        
        logger.info(f"Enhanced model initialized with window length {self.cache_config['window_length']}")

    def _init_cuda_graphs(self) -> None:
        """Initialize CUDA graphs for optimized inference."""
        if not hasattr(self, "cuda_graphs"):
            self.cuda_graphs = {}
            
        try:
            # Create static input shapes for common operations
            static_shapes = [(1, 32), (1, 64), (1, 128), (1, 256)]
            
            for shape in static_shapes:
                # Capture forward pass graph
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    sample_input = torch.zeros(shape, dtype=torch.long).cuda()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        self.forward(sample_input)
                    self.cuda_graphs[shape] = g
                torch.cuda.current_stream().wait_stream(s)
                
            logger.debug("CUDA graphs initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize CUDA graphs: {e}")
            self.model_config.use_cuda_graphs = False

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Union[FloatTensor, LongTensor, Optional[List[Tuple[FloatTensor]]], Optional[Tensor]]]:
        """Prepare inputs for generation with optimized caching.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Optional cached key/value states
            attention_mask: Optional attention mask
            inputs_embeds: Optional input embeddings
            cache_position: Optional cache positions
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of prepared inputs
        """
        logger.debug("Preparing inputs for generation")
        
        # Try to use CUDA graph if available
        if (
            self.model_config.use_cuda_graphs 
            and input_ids.shape in self.cuda_graphs 
            and past_key_values is None
        ):
            logger.debug(f"Using CUDA graph for shape {input_ids.shape}")
            return self.cuda_graphs[input_ids.shape].replay()
        
        # Check for static cache
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(self.model.layers[0].self_attn, "past_key_value", None)
            has_static_cache = past_key_values is not None
            logger.debug(f"Using static cache: {has_static_cache}")

        # Calculate past length and handle cache
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, (Cache, DynamicCache)):
                logger.debug("Processing enhanced cache")
                past_length = cache_position[0] if cache_position is not None else 0
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_key_values.get_seq_length()
            else:
                logger.debug("Processing standard cache")
                past_length = cache_position[0] if cache_position is not None else 0
                cache_length = past_key_values[0].shape[2]
                max_cache_length = None

            # Handle input adjustments
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                logger.debug("Adjusting input_ids based on attention mask")
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            # Handle cache length limits
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                logger.debug("Truncating attention mask to max cache length")
                attention_mask = attention_mask[:, -max_cache_length:]

        # Generate position IDs if needed
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            logger.debug("Generating position IDs from attention mask")
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # Prepare model inputs
        if inputs_embeds is not None and past_key_values is None:
            logger.debug("Using input embeddings")
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            logger.debug("Using input IDs")
            model_inputs = {"input_ids": input_ids.contiguous()}

        # Handle cache positions
        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            logger.debug("Generating cache positions")
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        else:
            cache_position = cache_position[-input_length:]

        # Clear static cache if needed
        if has_static_cache:
            logger.debug("Clearing static cache")
            past_key_values = None

        # Update model inputs
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        
        logger.debug("Input preparation complete")
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with enhanced features.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key/value states
            inputs_embeds: Input embeddings
            labels: Optional labels for loss computation
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        logger.debug("Starting forward pass")
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = self._init_cache()

        # Call parent's forward
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        logger.debug("Forward pass complete")
        return outputs

    def _init_cache(self) -> Optional[List[Tuple[torch.FloatTensor]]]:
        """Initialize cache for the model.
        
        Returns:
            Initialized cache
        """
        return DynamicCache(
            initial_length=self.cache_config["window_length"],
            max_length=self.cache_config["max_length"]
        )

    def train_model(
        self,
        train_dataset: Union[ConcatDataset, str, Path, List[Union[str, Path]]],
        eval_dataset: Optional[Union[ConcatDataset, str, Path, List[Union[str, Path]]]] = None,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 100,
        logging_steps: int = 10,
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Train the model using various data sources.
        
        This method supports multiple data input formats:
        1. Single file: Pass a path to a single data file
        2. Multiple files: Pass a list of file paths
        3. Directory: Pass a directory path containing data files
        4. Pre-processed dataset: Pass a ConcatDataset instance
        
        The method automatically handles:
        - Streaming for large datasets
        - Batch processing for multiple files
        - Memory-efficient data loading
        - Distributed training when available
        
        Args:
            train_dataset: Training data source (file, directory, or dataset)
            eval_dataset: Optional evaluation data source
            batch_size: Training batch size
            gradient_accumulation_steps: Steps before gradient update
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for clipping
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            output_dir: Directory to save model and logs
            **kwargs: Additional training arguments
            
        Examples:
            # Train from a single file
            model.train_model("data/training.jsonl")
            
            # Train from multiple files
            model.train_model(["data/part1.jsonl", "data/part2.jsonl"])
            
            # Train from a directory
            model.train_model("data/training_files/")
            
            # Train with a pre-processed dataset
            model.train_model(processed_dataset)
        """
        logger.info("Starting model training")
        
        try:
            # Prepare training configuration
            train_config.update({
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "save_steps": save_steps,
                "eval_steps": eval_steps,
                "logging_steps": logging_steps,
                "output_dir": output_dir or "outputs",
                **kwargs
            })
            
            # Handle different data source types
            train_dataloader = self._prepare_dataloader(train_dataset, train_config)
            eval_dataloader = self._prepare_dataloader(eval_dataset, train_config) if eval_dataset else None
            
            # Initialize distributed training if available
            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs for distributed training")
                self.model = torch.nn.DataParallel(self.model)
            
            # Train model
            train(
                model=self,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                tokenizer=None,  # Will be set by train_config
                **train_config
            )
            
            logger.info("Model training complete")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _prepare_dataloader(
        self,
        data_source: Optional[Union[ConcatDataset, str, Path, List[Union[str, Path]]]],
        config: Dict[str, Any]
    ) -> Optional[torch.utils.data.DataLoader]:
        """Prepare data loader for training.
        
        Handles different data source types:
        1. ConcatDataset: Use directly
        2. Single file: Load and process
        3. Multiple files: Load and concatenate
        4. Directory: Load all files
        
        Args:
            data_source: Data source to prepare
            config: Training configuration
            
        Returns:
            Prepared DataLoader or None
        """
        if data_source is None:
            return None
            
        if isinstance(data_source, ConcatDataset):
            return get_dataloader(data_source, config)
            
        # Convert to Path object
        if isinstance(data_source, str):
            data_source = Path(data_source)
            
        # Handle directory
        if isinstance(data_source, Path) and data_source.is_dir():
            data_files = list(data_source.glob("*.jsonl"))
            logger.info(f"Found {len(data_files)} data files in directory")
            data_source = data_files
            
        # Handle file list
        if isinstance(data_source, list):
            datasets = []
            for file_path in data_source:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Data file not found: {file_path}")
                    
                # Use streaming for large files
                if file_path.stat().st_size > 1e9:  # 1GB
                    logger.info(f"Using streaming for large file: {file_path}")
                    dataset = self._create_streaming_dataset(file_path)
                else:
                    dataset = self._load_dataset(file_path)
                datasets.append(dataset)
            
            # Combine datasets
            combined_dataset = ConcatDataset(datasets)
            return get_dataloader(combined_dataset, config)
            
        # Handle single file
        if isinstance(data_source, Path):
            if not data_source.exists():
                raise FileNotFoundError(f"Data file not found: {data_source}")
                
            # Use streaming for large files
            if data_source.stat().st_size > 1e9:  # 1GB
                logger.info(f"Using streaming for large file: {data_source}")
                dataset = self._create_streaming_dataset(data_source)
            else:
                dataset = self._load_dataset(data_source)
            return get_dataloader(dataset, config)
            
        raise ValueError(f"Unsupported data source type: {type(data_source)}")

    def _create_streaming_dataset(self, file_path: Path) -> ConcatDataset:
        """Create streaming dataset for large files.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Streaming dataset
        """
        from datasets import load_dataset
        
        # Use Hugging Face datasets streaming
        streaming_dataset = load_dataset(
            "json",
            data_files=str(file_path),
            streaming=True
        )
        
        return ConcatDataset([streaming_dataset])

    def _load_dataset(self, file_path: Path) -> ConcatDataset:
        """Load dataset from file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded dataset
        """
        from datasets import load_dataset
        
        # Load dataset normally
        dataset = load_dataset(
            "json",
            data_files=str(file_path)
        )
        
        return ConcatDataset([dataset])

    def save_pretrained(
        self,
        save_directory: str,
        save_config: bool = True,
        **kwargs: Any
    ) -> None:
        """Save model with enhanced features.
        
        Args:
            save_directory: Directory to save model
            save_config: Whether to save configuration
            **kwargs: Additional saving arguments
        """
        logger.info(f"Saving model to {save_directory}")
        
        try:
            # Save model using llama-recipes utilities
            save_model(
                model=self,
                save_directory=save_directory,
                **kwargs
            )
            
            # Save additional configurations if needed
            if save_config:
                config_path = Path(save_directory) / "model_config.json"
                with open(config_path, "w") as f:
                    json.dump(
                        {
                            "model_config": asdict(self.model_config),
                            "cache_config": self.cache_config
                        },
                        f,
                        indent=2
                    )
            
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def __del__(self):
        """Cleanup when model is deleted."""
        logger.info(LogTemplates.MODEL_UNLOADED.format(
            model_name=self.model_name
        ))