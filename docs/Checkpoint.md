# Checkpoint Management

## Overview

LlamaHome's checkpoint management system provides robust functionality for saving and loading model states, tracking training progress, and managing training artifacts. The system supports both regular checkpointing and best model tracking with configurable save intervals and cleanup policies.

## Components

### CheckpointManager

The core component that handles checkpoint operations:

```python
from src.training.checkpoint import CheckpointManager, CheckpointConfig

config = CheckpointConfig(
    save_dir="checkpoints",
    save_steps=1000,
    save_epochs=True,
    keep_last_n=5,
    save_best=True,
    save_optimizer=True,
    save_scheduler=True,
    save_metrics=True,
    metric_for_best="loss",
    greater_is_better=False,
    save_format="safetensors"
)

checkpoint_manager = CheckpointManager(config)
```

## Features

### 1. Regular Checkpointing

Save checkpoints at specified intervals:

```python
# Save checkpoint
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics={"loss": 0.5, "accuracy": 0.95},
    step=1000,
    epoch=1
)

# Load checkpoint
state = checkpoint_manager.load_checkpoint("checkpoint-1000")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
scheduler.load_state_dict(state["scheduler"])
```

### 2. Best Model Tracking

Track and save the best model based on metrics:

```python
# Update best checkpoint
is_best = checkpoint_manager.update_best_checkpoint(
    metrics={"loss": 0.5, "accuracy": 0.95},
    step=1000,
    epoch=1
)

# Load best checkpoint
best_state = checkpoint_manager.load_best_checkpoint()
```

### 3. Safe File Operations

Ensure checkpoint integrity:

- Atomic save operations
- Checksum verification
- Backup creation
- Corruption detection

### 4. Automatic Cleanup

Manage checkpoint storage:

- Keep last N checkpoints
- Remove old checkpoints
- Clean corrupted files
- Track disk usage

## Configuration

### Environment Variables

```bash
# Checkpoint settings
CHECKPOINT_DIR=checkpoints
CHECKPOINT_FORMAT=safetensors
CHECKPOINT_KEEP_LAST=5
```

### Configuration File

```toml
[checkpoint]
save_dir = "checkpoints"
save_steps = 1000
save_epochs = true
keep_last_n = 5
save_best = true
save_optimizer = true
save_scheduler = true
save_metrics = true
metric_for_best = "loss"
greater_is_better = false
save_format = "safetensors"

[checkpoint.cleanup]
enabled = true
max_checkpoints = 10
min_free_space = "10GB"
```

## Directory Structure

```text
checkpoints/
├── checkpoint-1000/
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── metadata.json
├── checkpoint-2000/
├── best/
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── metadata.json
└── checkpoint_history.json
```

## Best Practices

### 1. Save Strategy

- Set appropriate save intervals
- Enable best model tracking
- Save all necessary states
- Use safe file operations

### 2. Storage Management

- Configure cleanup policies
- Monitor disk usage
- Implement backup strategy
- Handle corrupted files

### 3. Performance

- Use efficient save format
- Optimize save frequency
- Enable async operations
- Monitor I/O performance

## Troubleshooting

### Common Issues

1. Disk Space
   - Enable automatic cleanup
   - Reduce save frequency
   - Monitor disk usage
   - Remove unnecessary files

2. Corruption
   - Use safe file operations
   - Enable checksums
   - Implement backups
   - Verify loaded states

3. Performance
   - Optimize save frequency
   - Use efficient formats
   - Enable async saves
   - Monitor I/O

## API Reference

### CheckpointManager

```python
class CheckpointManager:
    def __init__(self, config: CheckpointConfig):
        """Initialize checkpoint manager with configuration."""
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> str:
        """Save checkpoint and return checkpoint path."""
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint state dict."""
        
    def update_best_checkpoint(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> bool:
        """Update best checkpoint if metrics improved."""
        
    def load_best_checkpoint(self) -> Dict[str, Any]:
        """Load best checkpoint state dict."""
        
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints based on policy."""
```

## Future Extensions

1. Checkpoint Features
   - Distributed checkpointing
   - Incremental saves
   - Checkpoint merging
   - State verification

2. Performance
   - Async operations
   - Compression
   - Partial loading
   - Streaming saves

3. Integration
   - Cloud storage
   - Version control
   - Metadata tracking
   - Visualization tools 