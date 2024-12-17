# Agent Instructions

## System Context

LlamaHome is a hybrid implementation combining llama-recipes and H2O features for optimal performance. The system follows these key principles:

1. Architecture Pattern
   - Modular core components with clear separation of concerns
   - Hybrid attention mechanisms combining multiple approaches
   - Comprehensive caching and resource management
   - Asynchronous operations for optimal performance

2. Implementation Standards
   - Python 3.11 (3.13 not supported)
   - Async/await for I/O operations
   - Type hints and comprehensive docstrings
   - Singleton pattern for managers
   - Dependency injection where appropriate

3. Core Components
   - Model management with hybrid features
   - Training pipeline with optimized processing
   - Configuration system with hierarchical structure
   - Caching system with LRU implementation

## Current State

1. Implemented Features
   - Core model management system
   - Basic training pipeline
   - Configuration management
   - Logging system
   - Cache management
   - CLI interface

2. In Progress
   - Advanced attention mechanisms
   - Resource optimization
   - Performance monitoring
   - Security enhancements

3. Next Steps
   - Complete hybrid attention implementation
   - Enhance training pipeline
   - Implement advanced caching
   - Add comprehensive monitoring

## Implementation Patterns

1. Model Management

   ```python
   class ModelManager:
       _instance = None
       
       def __new__(cls) -> 'ModelManager':
           if cls._instance is None:
               cls._instance = super().__new__(cls)
           return cls._instance
       
       async def initialize_model(self, config: Dict[str, Any]) -> Model:
           """Initialize model with hybrid features."""
           pass
   ```

2. Attention Mechanism

   ```python
   class HybridAttention:
       """Combines llama-recipes and H2O attention features."""
       
       async def compute_attention(
           self,
           input_tensor: torch.Tensor,
           mask: Optional[torch.Tensor] = None
       ) -> torch.Tensor:
           pass
   ```

3. Training Pipeline

   ```python
   class TrainingPipeline:
       """Implements hybrid training approach."""
       
       async def train(
           self,
           data: Dataset,
           config: TrainingConfig
       ) -> TrainingResult:
           pass
   ```

## Key Considerations

1. Performance
   - Use async/await for I/O operations
   - Implement proper caching strategies
   - Monitor resource utilization
   - Optimize memory usage
   - Handle large datasets efficiently

2. Error Handling
   - Use specific exception types
   - Implement proper cleanup
   - Log errors with context
   - Handle edge cases
   - Validate inputs

3. Security
   - Validate all inputs
   - Secure configuration handling
   - Proper resource cleanup
   - Access control implementation
   - Audit logging

## Directory Structure

```text
src/
├── core/
│   ├── attention.py     # Hybrid attention mechanisms
│   ├── model.py         # Core model implementation
│   ├── cache.py         # Caching system
│   └── config_handler.py # Configuration management
├── training/
│   ├── pipeline.py      # Training orchestration
│   ├── data.py         # Data management
│   ├── resources.py    # Resource handling
│   └── optimization.py # Training optimization
└── utils/
    ├── log_manager.py  # Singleton logging
    ├── model_manager.py # Model lifecycle
    └── cache_manager.py # Cache management
```

## Configuration System

1. Hierarchy
   - Base configurations in `.config/`
   - Environment-specific overrides
   - Runtime parameters
   - Command-line arguments

2. Files
   - `.config/models.json` - Model configurations
   - `.config/training_config.yaml` - Training parameters
   - `.env` - Environment variables
   - `pyproject.toml` - Project settings

## Testing Strategy

1. Test Types
   - Unit tests for components
   - Integration tests for workflows
   - Performance benchmarks
   - Security tests

2. Coverage Requirements
   - Core components: 90%+
   - Utilities: 85%+
   - Integration points: 80%+

## Documentation Requirements

1. Code Documentation
   - Comprehensive docstrings
   - Type hints
   - Implementation notes
   - Performance considerations

2. Architecture Documentation
   - Component interactions
   - Data flow diagrams
   - Configuration details
   - Security considerations

## Development Workflow

1. Branch Strategy
   - feature/* for new features
   - bugfix/* for bug fixes
   - perf/* for optimizations
   - docs/* for documentation

2. Commit Messages

   ```text
   type(scope): concise description
   
   - Implementation details
   - Design decisions
   - Performance implications
   ```

## Critical Paths

1. Model Processing

   ```python
   async def process_input(
       text: str,
       config: Dict[str, Any],
       cache_manager: Optional[CacheManager] = None
   ) -> str:
       """Process input with caching support."""
       pass
   ```

2. Training Flow

   ```python
   async def train_model(
       config: TrainingConfig,
       data: Dataset,
       callbacks: List[Callback]
   ) -> TrainingResult:
       """Train model with monitoring."""
       pass
   ```

3. Resource Management

   ```python
   async def manage_resources(
       config: ResourceConfig,
       monitoring: MonitoringSystem
   ) -> None:
       """Manage system resources."""
       pass
   ```

## Integration Points

1. Model Integration
   - llama-recipes model management
   - H2O optimization features
   - Custom attention mechanisms
   - Resource monitoring

2. Training Integration
   - Data pipeline
   - Optimization strategies
   - Progress monitoring
   - Resource management

3. System Integration
   - Configuration management
   - Logging system
   - Cache management
   - Security controls

## Next Actions

1. Immediate Tasks
   - Complete hybrid attention implementation
   - Enhance training pipeline
   - Implement advanced caching
   - Add comprehensive monitoring

2. Upcoming Features
   - Advanced resource optimization
   - Enhanced security controls
   - Extended monitoring capabilities
   - Additional model support

3. Future Considerations
   - Scalability improvements
   - Additional integrations
   - Performance optimizations
   - Security enhancements

## Handoff Notes

1. Current Focus
   - Implementing hybrid attention mechanisms
   - Optimizing training pipeline
   - Enhancing resource management
   - Improving monitoring system

2. Known Issues
   - Memory optimization needed
   - Cache invalidation refinement
   - Resource cleanup enhancement
   - Security hardening required

3. Priority Areas
   - Performance optimization
   - Resource management
   - Security implementation
   - Documentation updates

Remember to maintain:

- Clean code principles
- Comprehensive documentation
- Proper error handling
- Performance optimization
- Security considerations
