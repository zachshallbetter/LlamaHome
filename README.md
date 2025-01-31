# LlamaHome

A comprehensive training and inference pipeline for LLM models with efficient resource management and monitoring.

## Features

- 🚀 Efficient training pipeline with distributed support
- 📊 Advanced monitoring and metrics tracking
- 💾 Smart caching system with multiple backends
- 🔄 Checkpoint management with safetensors support
- 📈 Dynamic batch sizing and memory optimization
- 🛠️ Modular architecture with clean separation of concerns

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/zachshallbetter/LlamaHome.git
cd LlamaHome
```

2. Set up your environment:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Configure your environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Core Components

- **Training Pipeline**: Distributed training with efficient resource utilization
- **Data Management**: Streaming datasets with memory-efficient processing
- **Monitoring**: Real-time metrics tracking and visualization
- **Caching**: Multi-tier caching system with configurable policies
- **Checkpointing**: Robust model and training state management

## Documentation

Detailed documentation is available in the `docs` directory:

- [Architecture Overview](docs/Architecture.md)
- [Setup Guide](docs/Setup.md)
- [Training Guide](docs/Training.md)
- [Inference Guide](docs/Inference.md)
- [API Reference](docs/API.md)
- [Configuration](docs/Config.md)
- [Performance Tuning](docs/Performance.md)
- [Troubleshooting](docs/Troubleshooting.md)

## Example Usage

```python
from src.training.pipeline import TrainingPipeline
from src.training.checkpoint import CheckpointManager
from src.training.monitoring import MonitorManager

# Initialize components
pipeline = TrainingPipeline(config)
checkpoint_manager = CheckpointManager(config, "my_model")
monitor = MonitorManager(config, "my_model")

# Train model
pipeline.train(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    checkpoint_manager=checkpoint_manager,
    monitor=monitor
)
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Transformers
- Inspired by best practices in ML engineering
- Thanks to all contributors
