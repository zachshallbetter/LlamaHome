"""Platform-agnostic efficient attention implementation."""

import logging
import warnings
import platform
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union
import pkg_resources

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

XFORMERS_MIN_VERSION = "0.0.29.post1"


def _try_import_xformers():
    """Try to import xformers and its components.
    
    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
    """
    try:
        import xformers
        import xformers.ops
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error importing xformers: {str(e)}"


def _is_supported_platform() -> Tuple[bool, Optional[str]]:
    """Check if the current platform supports xformers.
    
    Returns:
        Tuple[bool, str]: (is_supported, reason)
    """
    system = platform.system()
    arch = platform.machine()
    
    if system == "Darwin":
        return False, "xformers is not supported on macOS"
    elif system == "Windows":
        return False, "xformers is not officially supported on Windows"
    elif system != "Linux":
        return False, f"xformers is not supported on {system}"
    
    # Check Python version
    import sys
    if sys.version_info < (3, 9):
        return False, "xformers requires Python 3.9 or higher"
        
    return True, None


def _check_xformers_version() -> Tuple[bool, Optional[str]]:
    """Check if installed xformers version meets minimum requirements.
    
    Returns:
        Tuple[bool, str]: (version_ok, reason)
    """
    is_supported, platform_reason = _is_supported_platform()
    if not is_supported:
        return False, platform_reason
        
    try:
        import xformers
        installed_version = pkg_resources.get_distribution("xformers").version
        version_ok = pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(XFORMERS_MIN_VERSION)
        if not version_ok:
            return False, f"xformers version must be >= {XFORMERS_MIN_VERSION}, got {installed_version}"
        return True, None
    except (ImportError, pkg_resources.DistributionNotFound):
        return False, "xformers is not installed"
    except Exception as e:
        return False, f"Error checking xformers version: {str(e)}"


@lru_cache(maxsize=1)
def get_optimal_attention_backend() -> Tuple[str, Optional[str]]:
    """Determine the best available attention implementation for the current environment.
    
    Returns:
        Tuple of (backend_name, reason)
    """
    # Try xformers first
    version_ok, version_reason = _check_xformers_version()
    if version_ok:
        import_ok, import_error = _try_import_xformers()
        if import_ok:
            return "xformers", None
        else:
            logger.warning(f"Failed to import xformers: {import_error}")
    else:
        logger.info(f"xformers not available: {version_reason}")

    # Fallback options in order of preference
    if torch.cuda.is_available():
        return "cuda", "Using CUDA implementation"
    elif torch.backends.mps.is_available():
        return "mps", "Using MPS implementation"
    else:
        return "cpu", "Using CPU implementation"


class EfficientAttention:
    """Platform-agnostic efficient attention implementation.
    Automatically selects the best backend based on hardware availability.
    """

    def __init__(
        self,
        device_map: Optional[str] = "auto",
        attention_dropout: float = 0.0,
        attention_mode: Optional[str] = None,
        mem_efficient: bool = True,
        force_backend: Optional[str] = None
    ):
        """Initialize efficient attention.
        
        Args:
            device_map: Device mapping strategy
            attention_dropout: Attention dropout probability
            attention_mode: Optional specific attention mode to use
            mem_efficient: Whether to use memory efficient attention when possible
            force_backend: Optional backend to force use ("xformers", "cuda", "mps", "cpu")
        """
        self.device_map = device_map
        self.attention_dropout = attention_dropout
        self.mem_efficient = mem_efficient
        self.attention_mode = attention_mode
        
        # Initialize backend
        if force_backend:
            success = self._init_backend(force_backend)
            if not success:
                raise ValueError(f"Failed to initialize forced backend: {force_backend}")
        else:
            self.backend, reason = get_optimal_attention_backend()
            if reason:
                logger.info(reason)
            
            # Initialize the selected backend
            success = self._init_backend(self.backend)
            if not success:
                # Fallback to standard attention
                self.backend = "cpu"
                logger.warning("Falling back to CPU implementation")

        # Validate configuration
        self._validate_config()

    def _init_backend(self, backend: str) -> bool:
        """Initialize specific attention backend.
        
        Args:
            backend: Backend to initialize
            
        Returns:
            bool: Whether initialization was successful
        """
        try:
            if backend == "xformers":
                from xformers.ops import memory_efficient_attention, fmha
                self.memory_efficient_attention = memory_efficient_attention
                self.fmha = fmha
                logger.info("Successfully initialized xformers attention")
            elif backend in ["cuda", "mps", "cpu"]:
                # Standard attention implementations don't need special initialization
                pass
            else:
                raise ValueError(f"Unknown backend: {backend}")
            
            self.backend = backend
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {backend} backend: {e}")
            return False

    def _validate_config(self) -> None:
        """Validate attention configuration."""
        if self.attention_dropout < 0 or self.attention_dropout > 1:
            raise ValueError(f"Attention dropout must be between 0 and 1, got {self.attention_dropout}")

        if self.backend == "xformers" and not _check_xformers_version():
            raise ValueError(
                f"xformers version must be >= {XFORMERS_MIN_VERSION}, "
                f"got {pkg_resources.get_distribution('xformers').version}"
            )

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: Optional[float] = None,
        scale: Optional[float] = None,
        causal: bool = False,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute attention with the most efficient available backend.

        Args:
            query: Query tensor (batch_size, seq_len, num_heads, head_dim)
            key: Key tensor (batch_size, seq_len, num_heads, head_dim)
            value: Value tensor (batch_size, seq_len, num_heads, head_dim)
            mask: Optional attention mask
            dropout_p: Optional dropout probability (overrides instance setting)
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
            causal: Whether to use causal attention
            **kwargs: Additional backend-specific arguments

        Returns:
            Output tensor (batch_size, seq_len, num_heads, head_dim)
            
        Raises:
            ValueError: If inputs have invalid shapes
            RuntimeError: If attention computation fails
        """
        try:
            # Input validation
            if not all(x.dim() == 4 for x in (query, key, value)):
                raise ValueError("Query, key and value must be 4-dimensional")

            if not query.shape[-1] == key.shape[-1] == value.shape[-1]:
                raise ValueError("Query, key and value must have same head dimension")

            # Set defaults
            dropout_p = dropout_p if dropout_p is not None else self.attention_dropout
            if scale is None:
                scale = query.shape[-1] ** -0.5

            # Handle different backends
            if self.backend == "xformers" and self.mem_efficient:
                try:
                    # Prepare inputs for xformers
                    attn_bias = None
                    if mask is not None:
                        attn_bias = torch.zeros_like(mask, dtype=query.dtype)
                        attn_bias = attn_bias.masked_fill(mask == 0, float("-inf"))
                    elif causal:
                        attn_bias = self.fmha.BlockDiagonalCausalMask()

                    return self.memory_efficient_attention(
                        query, key, value,
                        attn_bias=attn_bias,
                        p=dropout_p,
                        scale=scale,
                        **kwargs
                    )
                except Exception as e:
                    logger.warning(f"xformers attention failed, falling back to standard: {e}")

            # Standard scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

            # Apply mask
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
            elif causal:
                seq_len = query.size(-2)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
                    diagonal=1
                )
                attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

            # Compute attention weights
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply dropout
            if dropout_p > 0:
                attention_weights = F.dropout(attention_weights, p=dropout_p)

            # Compute output
            return torch.matmul(attention_weights, value)

        except Exception as e:
            raise RuntimeError(f"Attention computation failed: {e}")

    def set_backend(self, backend: str) -> None:
        """Manually set the attention backend.
        
        Args:
            backend: Backend to use ('xformers', 'cuda', 'mps', or 'cpu')
            
        Raises:
            ValueError: If backend is not supported
        """
        if backend not in ["xformers", "cuda", "mps", "cpu"]:
            raise ValueError(f"Unsupported backend: {backend}")
        
        if backend == "xformers":
            if not _check_xformers_version():
                raise ValueError(
                    f"xformers version must be >= {XFORMERS_MIN_VERSION} to use xformers backend"
                )
            try:
                import xformers.ops
                self.backend = "xformers"
            except ImportError:
                raise ValueError("xformers not available")
        else:
            self.backend = backend

        logger.info(f"Switched to {self.backend} attention backend")


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention layer with efficient backend."""

    def __init__(
        self,
        config: Dict[str, Any],
        attention_dropout: float = 0.0,
        causal: bool = False
    ):
        """Initialize multi-head attention.
        
        Args:
            config: Configuration dictionary
            attention_dropout: Attention dropout probability
            causal: Whether to use causal attention
        """
        super().__init__()
        self.device_map = config.get("device_map", "auto")
        self.attention = EfficientAttention(
            device_map=self.device_map,
            attention_dropout=attention_dropout,
            mem_efficient=config.get("mem_efficient", True),
            force_backend=config.get("force_backend")
        )
        self.causal = causal

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            **kwargs: Additional arguments passed to attention implementation
            
        Returns:
            Output tensor
        """
        return self.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            causal=self.causal,
            **kwargs
        )


class AttentionError(Exception):
    """Base class for attention-related errors."""
    pass


class BackendError(AttentionError):
    """Error related to attention backend."""
    pass


class ShapeError(AttentionError):
    """Error related to tensor shapes."""
    pass
