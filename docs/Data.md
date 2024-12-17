# Data Management and Training

This document outlines the data management and training infrastructure in LlamaHome.

## Overview

The data management system provides comprehensive handling of training data with:

- Robust validation and metrics tracking
- Efficient batch processing with progress monitoring
- Train/validation split functionality
- Early stopping and checkpointing
- LoRA fine-tuning support

## Components

### TrainingData

The core component for managing training data:

```python
class TrainingData:
    """Manages training data processing and storage."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 4,
        max_workers: int = 4,
        config: Optional[Dict] = None
    ) -> None:
        """Initialize training data manager."""
```

### ProgressCallback

Tracks training progress with rich progress bars:

```python
class ProgressCallback(TrainerCallback):
    """Custom callback for training progress."""
    
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        )
```

### ConversationDataset

Handles conversation data formatting:

```python
class ConversationDataset(Dataset):
    """Dataset for conversation samples."""
    
    def __init__(
        self,
        conversations: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """Initialize dataset."""
```

## Configuration

### Training Configuration

Configuration is managed through `.config/training_config.yaml`:

```yaml
training:
  # General settings
  batch_size: 4
  max_workers: 4
  max_length: 512
  validation_split: 0.1

  # Training parameters
  epochs: 3
  learning_rate: 5e-5
  warmup_steps: 100
  weight_decay: 0.01
  gradient_accumulation_steps: 4
  fp16: true
  
  # LoRA settings
  lora:
    r: 8
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
    bias: "none"
    
  # Early stopping
  early_stopping:
    enabled: true
    patience: 3
    min_delta: 0.01
```

### Model-Specific Overrides

Models can have specific configuration overrides:

```yaml
model_configs:
  llama:
    batch_size: 2
    max_length: 1024
    lora:
      r: 16
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Data Processing

### Validation Pipeline

1. Sample Validation
   - JSONL format validation
   - Conversation structure checking
   - Content validation
   - Length constraints

2. Dataset Validation
   - Train/validation splitting
   - Tokenization verification
   - Special token handling
   - Metrics calculation

### Batch Processing

Features:

- Configurable batch sizes
- Progress tracking with rich progress bars
- Async processing
- Resource management
- Early stopping support

Implementation:

```python
async def process_samples(
    self,
    model_name: str = "llama",
    model_version: Optional[str] = None
) -> None:
    """Process and prepare training samples."""
    with Progress() as progress:
        load_task = progress.add_task(
            "Loading samples...",
            total=len(sample_files)
        )
```

## Training Process

### Data Preparation

1. Load and validate samples
2. Split into train/validation sets
3. Apply tokenization
4. Create DataLoader instances

### Model Setup

1. Load base model
2. Configure LoRA parameters
3. Prepare for training
4. Set up callbacks

### Training Loop

1. Initialize progress tracking
2. Train with validation
3. Apply early stopping
4. Save checkpoints

### Model Saving

1. Save model weights
2. Save tokenizer
3. Save training metrics
4. Save configuration

## Metrics and Monitoring

### Training Metrics

Metrics are saved in JSON format:

```json
{
  "train_loss": 1.234,
  "eval_loss": 1.123,
  "learning_rate": 5e-5,
  "epoch": 1,
  "step": 100
}
```

### Progress Tracking

1. Overall Progress:
   - Total epochs
   - Total steps
   - Time remaining

2. Current Progress:
   - Current epoch
   - Current step
   - Batch progress

## Best Practices

### Data Organization

1. Directory Structure:

   ```text
   data/
   ├── training/
   │   ├── samples/
   │   ├── processed/
   │   └── checkpoints/
   ├── models/
   │   └── finetuned/
   └── metrics/
   ```

### Resource Management

1. Memory:
   - Batch size optimization
   - Gradient accumulation
   - Mixed precision training
   - Memory-efficient training

2. Processing:
   - Multi-worker data loading
   - GPU utilization
   - Progress monitoring
   - Resource cleanup

## Usage Examples

### Basic Training Setup

```python
# Initialize manager
manager = TrainingData(
    data_dir=Path("data"),
    batch_size=4,
    max_workers=4
)

# Process samples
await manager.process_samples(
    model_name="llama",
    model_version="3.3-7b"
)

# Train model
await manager.train_model(
    model_name="llama",
    model_version="3.3-7b"
)
```

### Custom Configuration

```python
config = {
    "batch_size": 2,
    "max_length": 1024,
    "lora": {
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
}

manager = TrainingData(
    data_dir=Path("data"),
    config=config
)
```

## Testing

### Test Coverage

1. Data Tests:
   - Sample loading
   - Validation splitting
   - Tokenization
   - Batch processing

2. Training Tests:
   - Configuration loading
   - Progress tracking
   - Early stopping
   - Metric saving

Example test:

```python
@pytest.mark.asyncio
async def test_process_samples():
    manager = TrainingData(tmp_path)
    await manager.process_samples("llama", "3.3-7b")
    assert (tmp_path / "processed").exists()
```
