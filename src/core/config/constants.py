"""Constants and configuration values for LlamaHome.

This module defines shared constants, environment variables, and configuration 
values used across the LlamaHome project.
"""

from pathlib import Path
from typing import Dict, List

# Model configuration
SUPPORTED_MODELS = {
    "llama3.3": {
        "name": "Llama 3.3",
        "requires_gpu": True,
        "min_gpu_memory": {"7b": 12, "13b": 24, "70b": 100},
        "h2o_config": {
            "enable": True,
            "window_length": 1024,
            "heavy_hitter_tokens": 256,
            "position_rolling": True,
            "max_sequence_length": 65536,
        },
        "env_vars": [
            "LLAMA_NUM_GPU_LAYERS",
            "LLAMA_MAX_SEQ_LEN",
            "LLAMA_MAX_BATCH_SIZE", 
            "LLAMA_H2O_ENABLED",
            "LLAMA_H2O_WINDOW_LENGTH",
            "LLAMA_H2O_HEAVY_HITTERS",
        ],
    },
    "gpt4": {
        "name": "GPT-4 Turbo",
        "requires_key": True,
        "max_tokens": 128000,
        "env_vars": ["LLAMAHOME_GPT4_API_KEY", "LLAMAHOME_GPT4_ORG_ID"],
    },
    "claude": {
        "name": "Claude 3",
        "requires_key": True,
        "model_variants": ["opus", "sonnet", "haiku"],
        "env_vars": ["LLAMAHOME_CLAUDE_API_KEY", "LLAMAHOME_CLAUDE_ORG_ID"],
    },
}

# Environment variables
ENV_VARS = [
    "LLAMA_MODEL",
    "LLAMA_MODEL_SIZE",
    "LLAMA_MODEL_VARIANT",
    "LLAMA_NUM_GPU_LAYERS",
    "LLAMA_MAX_SEQ_LEN",
    "LLAMA_MAX_BATCH_SIZE",
    "LLAMA_H2O_ENABLED", 
    "LLAMA_H2O_WINDOW_LENGTH",
    "LLAMA_H2O_HEAVY_HITTERS",
    "LLAMAHOME_GPT4_API_KEY",
    "LLAMAHOME_GPT4_ORG_ID",
    "LLAMAHOME_CLAUDE_API_KEY",
    "LLAMAHOME_CLAUDE_ORG_ID",
]
