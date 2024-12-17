# LlamaHome Core Architecture

This document outlines the core architecture and integration patterns of the LlamaHome system.

## System Overview

LlamaHome's core architecture consists of several key components that work together to provide a robust and scalable system:

- Async request handling and queue management
- Model integration with H2O optimization
- Data storage and management
- Training integration
- GUI integration
- Comprehensive error handling

## Core Components and Boundaries

### Component Hierarchy

1. Core Layer (`src/core/`):
   - Request handling and queuing
   - Model inference and streaming
   - H2O optimization
   - Core error handling

2. Utility Layer (`utils/`):
   - Model setup and configuration
   - Environment management
   - Resource validation
   - Logging and monitoring

3. Interface Layer (`src/interfaces/`):
   - CLI/GUI implementations
   - API endpoints
   - User interaction

### Model Configuration Boundaries

1. Base Models:

   ```python
   class ModelConfig:
       """Base model configuration."""
       name: str
       requires_gpu: bool = False
       requires_key: bool = False
       env_vars: List[str] = []
   ```

2. LLM-Specific Configuration:

   ```python
   class LlamaConfig(ModelConfig):
       """Llama model configuration."""
       min_gpu_memory: Dict[str, int]
       h2o_config: H2OConfig
       
   class GPT4Config(ModelConfig):
       """GPT-4 configuration."""
       max_tokens: int
       api_key: str
       org_id: str

   class ClaudeConfig(ModelConfig):
       """Claude model configuration."""
       model_variants: List[str]
       api_key: str
       org_id: str
   ```

3. Environment Configuration:

   ```python
   class EnvConfig:
       """Environment configuration."""
       
       MODEL_ENV_VARS = {
           "llama": [
               "LLAMA_NUM_GPU_LAYERS",
               "LLAMA_MAX_SEQ_LEN",
               "LLAMA_MAX_BATCH_SIZE",
               "LLAMA_H2O_ENABLED",
               "LLAMA_H2O_WINDOW_LENGTH",
               "LLAMA_H2O_HEAVY_HITTERS",
           ],
           "gpt4": [
               "LLAMAHOME_GPT4_API_KEY",
               "LLAMAHOME_GPT4_ORG_ID",
           ],
           "claude": [
               "LLAMAHOME_CLAUDE_API_KEY",
               "LLAMAHOME_CLAUDE_ORG_ID",
           ],
       }
   ```

### Testing Infrastructure

1. Test Directory Structure:
   ```text
   tests/
   ├── core/              # Core functionality tests
   │   ├── test_model.py
   │   ├── test_request.py
   │   └── test_h2o.py
   ├── integration/       # Integration tests
   │   └── test_core_integration.py
   ├── data/             # Data management tests
   │   ├── test_storage.py
   │   └── test_training.py
   └── conftest.py       # Shared test fixtures
   ```

2. Test Fixtures:

   ```python
   @pytest.fixture(scope="session")
   def test_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
       """Create test data directory."""
       return tmp_path_factory.mktemp("test_data")

   @pytest.fixture(scope="session")
   def test_config_dir(test_data_dir: Path) -> Path:
       """Create test config directory."""
       config_dir = test_data_dir / "config"
       config_dir.mkdir(exist_ok=True)
       return config_dir

   @pytest.fixture(scope="session")
   def test_models_dir(test_data_dir: Path) -> Path:
       """Create test models directory."""
       models_dir = test_data_dir / "models"
       models_dir.mkdir(exist_ok=True)
       return models_dir
   ```

3. Environment Management:

   ```python
   @pytest.fixture(scope="function")
   def clean_env() -> None:
       """Provide clean environment for testing."""
       original_env = dict(os.environ)
       env_vars_to_clear = EnvConfig.MODEL_ENV_VARS["llama"] + \
                          EnvConfig.MODEL_ENV_VARS["gpt4"] + \
                          EnvConfig.MODEL_ENV_VARS["claude"]
       
       for var in env_vars_to_clear:
           os.environ.pop(var, None)
       
       yield
       
       os.environ.clear()
       os.environ.update(original_env)
   ```

### Resource Management

1. GPU Resource Management:

   ```python
   class GPUManager:
       """GPU resource management."""
       
       def __init__(self, model_config: ModelConfig):
           self.min_memory = model_config.min_gpu_memory
           self.current_usage = 0
           self._monitor = GPUMonitor()
       
       def check_requirements(self, model_size: str) -> bool:
           """Check GPU requirements for model size."""
           if not torch.cuda.is_available():
               return False
               
           available_memory = self._monitor.get_available_memory()
           required_memory = self.min_memory[model_size]
           return available_memory >= required_memory
   ```

2. API Resource Management:

   ```python
   class APIManager:
       """API resource management."""
       
       def __init__(self, model_config: ModelConfig):
           self.api_key = os.getenv(f"LLAMAHOME_{model_config.name}_API_KEY")
           self.org_id = os.getenv(f"LLAMAHOME_{model_config.name}_ORG_ID")
           self._rate_limiter = RateLimiter()
       
       async def validate_credentials(self) -> bool:
           """Validate API credentials."""
           if not self.api_key or not self.org_id:
               return False
           
           try:
               return await self._test_credentials()
           except APIError:
               return False
   ```

### State Transitions

1. Model State Transitions:

   ```python
   class ModelState(Enum):
       """Model states."""
       UNINITIALIZED = "uninitialized"
       LOADING = "loading"
       READY = "ready"
       PROCESSING = "processing"
       ERROR = "error"
       SHUTDOWN = "shutdown"

   class ModelStateManager:
       """Model state management."""
       
       def __init__(self):
           self._state = ModelState.UNINITIALIZED
           self._transitions = {
               ModelState.UNINITIALIZED: [ModelState.LOADING],
               ModelState.LOADING: [ModelState.READY, ModelState.ERROR],
               ModelState.READY: [ModelState.PROCESSING, ModelState.SHUTDOWN],
               ModelState.PROCESSING: [ModelState.READY, ModelState.ERROR],
               ModelState.ERROR: [ModelState.LOADING, ModelState.SHUTDOWN],
               ModelState.SHUTDOWN: [ModelState.LOADING],
           }
       
       async def transition(self, new_state: ModelState) -> bool:
           """Perform state transition."""
           if new_state in self._transitions[self._state]:
               await self._handle_transition(new_state)
               self._state = new_state
               return True
           return False
   ```

2. Request State Transitions:

   ```python
   class RequestState(Enum):
       """Request states."""
       PENDING = "pending"
       QUEUED = "queued"
       PROCESSING = "processing"
       COMPLETED = "completed"
       FAILED = "failed"

   class RequestStateManager:
       """Request state management."""
       
       def __init__(self):
           self._state = RequestState.PENDING
           self._transitions = {
               RequestState.PENDING: [RequestState.QUEUED, RequestState.FAILED],
               RequestState.QUEUED: [RequestState.PROCESSING, RequestState.FAILED],
               RequestState.PROCESSING: [RequestState.COMPLETED, RequestState.FAILED],
               RequestState.COMPLETED: [],
               RequestState.FAILED: [],
           }
       
       def can_transition(self, new_state: RequestState) -> bool:
           """Check if transition is valid."""
           return new_state in self._transitions[self._state]
   ```

### Performance Monitoring

1. Latency Monitoring:

   ```python
   class LatencyMonitor:
       """Monitor operation latency."""
       
       def __init__(self):
           self.thresholds = {
               "inference": 100,  # ms
               "queue": 10,      # ms
               "h2o": 50,       # ms
               "error": 5,      # ms
           }
           self._metrics = defaultdict(list)
       
       def record(self, operation: str, duration: float) -> None:
           """Record operation duration."""
           self._metrics[operation].append(duration)
           if duration > self.thresholds[operation]:
               logger.warning(f"{operation} exceeded threshold: {duration}ms")
   ```

2. Resource Monitoring:

   ```python
   class ResourceMonitor:
       """Monitor resource usage."""
       
       def __init__(self):
           self.thresholds = {
               "queue_size": 100,
               "batch_size": 32,
               "cache_size": 1024 * 1024 * 1024,  # 1GB
               "gpu_memory": 0.9,  # 90%
           }
           self._metrics = ResourceMetrics()
       
       def check_thresholds(self) -> List[str]:
           """Check resource thresholds."""
           violations = []
           if self._metrics.queue_size > self.thresholds["queue_size"]:
               violations.append("queue_size")
           if self._metrics.gpu_usage > self.thresholds["gpu_memory"]:
               violations.append("gpu_memory")
           return violations
   ```

### Error Handling

1. Error Hierarchy:

   ```python
   class LlamaHomeError(Exception):
       """Base error class."""
       pass

   class ModelError(LlamaHomeError):
       """Model-related errors."""
       pass

   class ResourceError(LlamaHomeError):
       """Resource-related errors."""
       pass

   class ConfigError(LlamaHomeError):
       """Configuration-related errors."""
       pass

   class APIError(LlamaHomeError):
       """API-related errors."""
       pass
   ```

2. Error Recovery:

   ```python
   class ErrorRecovery:
       """Error recovery strategies."""
       
       async def handle_error(self, error: LlamaHomeError) -> bool:
           """Handle error with appropriate strategy."""
           if isinstance(error, ModelError):
               return await self._recover_model()
           elif isinstance(error, ResourceError):
               return await self._recover_resources()
           elif isinstance(error, APIError):
               return await self._recover_api()
           return False
       
       async def _recover_model(self) -> bool:
           """Attempt model recovery."""
           try:
               await self.model.unload()
               await self.model.load()
               return True
           except Exception:
               return False
   ```

### Testing Strategy

1. Core Testing:

   ```python
   @pytest.mark.core
   class TestModelCore:
       """Core model tests."""
       
       @pytest.mark.asyncio
       async def test_model_lifecycle(self, clean_env):
           """Test model lifecycle."""
           model = LlamaIntegration()
           assert not model.is_loaded
           
           await model.load()
           assert model.is_loaded
           
           await model.unload()
           assert not model.is_loaded
       
       @pytest.mark.asyncio
       async def test_state_transitions(self, clean_env):
           """Test state transitions."""
           state_manager = ModelStateManager()
           assert await state_manager.transition(ModelState.LOADING)
           assert await state_manager.transition(ModelState.READY)
           assert not await state_manager.transition(ModelState.SHUTDOWN)
   ```

2. Resource Testing:

   ```python
   @pytest.mark.core
   class TestResourceCore:
       """Core resource tests."""
       
       def test_gpu_requirements(self, clean_env):
           """Test GPU requirements."""
           config = ModelConfig(min_gpu_memory={"13b": 24})
           manager = GPUManager(config)
           assert manager.check_requirements("13b") == torch.cuda.is_available()
       
       def test_resource_monitoring(self, clean_env):
           """Test resource monitoring."""
           monitor = ResourceMonitor()
           violations = monitor.check_thresholds()
           assert isinstance(violations, list)
   ```

### Documentation Standards

1. Component Documentation:

   ```python
   class Component:
       """
       Component description.

       Attributes:
           attr1: Description of attribute 1
           attr2: Description of attribute 2

       Methods:
           method1: Description of method 1
           method2: Description of method 2

       Raises:
           Error1: Description of error condition 1
           Error2: Description of error condition 2
       """
   ```

2. Configuration Documentation:

   ```yaml
   # model_config.yaml
   models:
     llama3.3:
       name: "Llama 3.3"
       requires_gpu: true
       min_gpu_memory:
         7b: 12    # GB
         13b: 24   # GB
         70b: 100  # GB
       h2o_config:
         enable: true
         window_length: 1024
         heavy_hitter_tokens: 256
         position_rolling: true
         max_sequence_length: 65536
   ```

### Performance Requirements

1. Operation Latency:
   - Model inference: < 100ms
   - Request queuing: < 10ms
   - H2O optimization: < 50ms
   - Error handling: < 5ms

2. Resource Usage:
   - Queue size: ≤ 100 requests
   - Batch size: ≤ 32 samples
   - Cache size: ≤ 1GB
   - GPU memory: ≤ 90% utilization

3. Reliability Targets:
   - Error rate: < 0.1%
   - Recovery success: > 99%
   - Availability: > 99.9%
   - Response success: > 99.5%

### Environment Setup Boundaries

1. Directory Structure:

   ```python
   REQUIRED_DIRS: List[Path] = [
       Path("logs"),
       Path(".config"),
       Path("models"),
       Path("data"),
       Path("tests/data"),
   ]
   ```

2. Environment Configuration:

   ```python
   # Core environment settings
   DEFAULT_ENV_CONFIG = {
       # Core settings
       "PYTHONPATH": "${PYTHONPATH}:${PWD}",
       "LLAMAHOME_ENV": "development",

       # Model settings
       "LLAMA_MODEL": "llama3.3",
       "LLAMA_MODEL_SIZE": "13b",
       "LLAMA_MODEL_VARIANT": "chat",
       "LLAMA_MODEL_QUANT": "q4_0",
       "LLAMA_NUM_GPU_LAYERS": "32",
       "LLAMA_MAX_SEQ_LEN": "32768",

       # Performance settings
       "LLAMAHOME_BATCH_SIZE": "1000",
       "LLAMAHOME_MAX_WORKERS": "4",
       "LLAMAHOME_CACHE_SIZE": "1024",
   }
   ```

3. Configuration Templates:

   ```python
   CONFIG_TEMPLATES: Dict[Path, str] = {
       Path(".config/model_config.yaml"): MODEL_CONFIG_TEMPLATE,
       Path(".config/code_check.yaml"): CODE_CHECK_TEMPLATE,
       Path(".config/llamahome.types.ini"): TYPE_CHECK_TEMPLATE,
   }
   ```

### Logging System

1. Logger Configuration:

   ```python
   class LoggerConfig:
       """Logger configuration."""
       
       PROFESSIONAL_THEME = Theme({
           "info": "cyan",
           "warning": "yellow",
           "error": "red bold",
           "section": "blue bold",
           "success": "green",
       })

       FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
       CONSOLE_FORMAT = "%(message)s"
   ```

2. Logger Factory:

   ```python
   def get_logger(
       name: str,
       log_file: Optional[Path] = None,
       level: str = "INFO"
   ) -> logging.Logger:
       """Create logger with professional formatting."""
       logger = logging.getLogger(name)
       logger.setLevel(getattr(logging, level.upper()))
       
       # Configure handlers
       console_handler = RichHandler(
           rich_tracebacks=True,
           markup=True,
           show_time=True,
           show_path=True,
           console=Console(theme=PROFESSIONAL_THEME)
       )
       
       return logger
   ```

3. System Logging:

   ```python
   def log_system_info() -> None:
       """Log system configuration."""
       system_info = {
           "Operating System": f"{platform.system()} ({platform.release()})",
           "Architecture": platform.machine(),
           "Python Version": platform.python_version(),
           "PyTorch Version": torch.__version__,
           "Compute Backend": get_compute_backend(),
       }
       log_section("System Configuration", system_info)
   ```

### Model Management

1. Model Configuration:

   ```python
   class ModelConfig(TypedDict, total=False):
       """Model configuration type definition."""
       name: str
       versions: List[str]
       variants: List[str]
       requires_gpu: bool
       min_gpu_memory: Dict[str, int]
       h2o_config: Dict[str, Any]
   ```

2. Model Manager:

   ```python
   class ModelManager:
       """Model configuration and setup."""
       
       DEFAULT_MODEL_DIR = Path.home() / ".llamahome" / "models"
       ENV_FILE = Path(".env")

       def __init__(self, config_path: Optional[Path] = None):
           self.config_path = config_path or Path(".config/model_config.yaml")
           self.config = self._load_config()
           self.h2o_manager = None
           load_dotenv()
   ```

3. H2O Integration:

   ```python
   def _setup_h2o(self, model_config: ModelConfig) -> None:
       """Configure H2O acceleration."""
       if h2o_config := model_config.get("h2o_config"):
           config = H2OConfig(
               enabled=True,
               window_length=get_window_length(),
               heavy_hitter_tokens=get_heavy_hitters(),
               max_sequence_length=get_max_seq_len(),
           )
           self.h2o_manager = H2OManager(config.__dict__)
   ```

### Utility Layer Boundaries

1. Setup Utilities:

   ```python
   class SetupManager:
       """Manage system setup process."""
       
       async def setup_environment(self) -> None:
           """Set up environment."""
           self._create_directories()
           self._configure_environment()
           self._setup_logging()
           await self._setup_model()

       def _create_directories(self) -> None:
           """Create required directories."""
           for directory in REQUIRED_DIRS:
               directory.mkdir(parents=True, exist_ok=True)
   ```

2. Resource Utilities:

   ```python
   class ResourceManager:
       """Manage system resources."""
       
       def check_requirements(self) -> None:
           """Verify system requirements."""
           self._check_python_version()
           self._check_gpu_requirements()
           self._check_disk_space()
           self._check_memory()
   ```

3. Configuration Utilities:

   ```python
   class ConfigManager:
       """Manage system configuration."""
       
       def load_config(self) -> None:
           """Load configuration files."""
           self._load_env_config()
           self._load_model_config()
           self._load_type_config()
   ```

### Integration Points

1. Environment Integration:

   ```python
   class EnvironmentIntegration:
       """Environment integration layer."""
       
       def __init__(self):
           self.setup_manager = SetupManager()
           self.resource_manager = ResourceManager()
           self.config_manager = ConfigManager()
       
       async def initialize(self) -> None:
           """Initialize environment."""
           self.resource_manager.check_requirements()
           self.config_manager.load_config()
           await self.setup_manager.setup_environment()
   ```

2. Logging Integration:

   ```python
   class LoggingIntegration:
       """Logging integration layer."""
       
       def __init__(self):
           self.logger = get_logger(__name__)
           self.console = Console(theme=PROFESSIONAL_THEME)
       
       def configure_logging(self) -> None:
           """Configure logging system."""
           self._setup_file_logging()
           self._setup_console_logging()
           self._log_system_info()
   ```

3. Model Integration:

   ```python
   class ModelIntegration:
       """Model integration layer."""
       
       def __init__(self):
           self.model_manager = ModelManager()
           self.h2o_manager = None
       
       async def setup_model(self, model_name: str) -> None:
           """Set up model environment."""
           await self.model_manager.setup_model(model_name)
           if self.model_manager.h2o_manager:
               self.h2o_manager = self.model_manager.h2o_manager
   ```

### Utility Layer Testing

1. Setup Testing:

   ```python
   @pytest.mark.utility
   class TestSetupUtilities:
       """Test setup utilities."""
       
       async def test_environment_setup(self, tmp_path: Path):
           """Test environment setup."""
           setup = SetupManager(base_path=tmp_path)
           await setup.setup_environment()
           
           # Verify directories
           for directory in REQUIRED_DIRS:
               assert (tmp_path / directory).exists()
   ```

2. Resource Testing:

   ```python
   @pytest.mark.utility
   class TestResourceUtilities:
       """Test resource utilities."""
       
       def test_resource_checks(self):
           """Test resource verification."""
           manager = ResourceManager()
           manager.check_requirements()
   ```

3. Configuration Testing:

   ```python
   @pytest.mark.utility
   class TestConfigUtilities:
       """Test configuration utilities."""
       
       def test_config_loading(self, tmp_path: Path):
           """Test configuration loading."""
           manager = ConfigManager(config_dir=tmp_path)
           manager.load_config()
   ```

### Performance Boundaries

1. Setup Performance:
   - Directory creation: < 100ms
   - Configuration loading: < 50ms
   - Environment setup: < 1s
   - Model setup: < 5s

2. Resource Usage:
   - Memory overhead: < 100MB
   - Disk usage: < 1GB
   - CPU usage: < 10%
   - File handles: < 100

3. Monitoring Metrics:
   - Setup success rate
   - Configuration load time
   - Resource check duration
   - Error frequency

### Errors

1. Setup Errors:

   ```python
   class SetupError(LlamaHomeError):
       """Setup-related errors."""
       pass

   class DirectoryError(SetupError):
       """Directory creation errors."""
       pass

   class ConfigError(SetupError):
       """Configuration errors."""
       pass
   ```

2. Recovery Strategies:

   ```python
   class SetupRecovery:
       """Setup recovery strategies."""
       
       async def recover_setup(self, error: SetupError) -> bool:
           """Attempt setup recovery."""
           if isinstance(error, DirectoryError):
               return await self._recover_directories()
           elif isinstance(error, ConfigError):
               return await self._recover_config()
           return False
   ```

### Directory Structure

1. Core Directories:

   ```python
   CORE_DIRS: List[Path] = [
       Path(".cache"),    # Application cache
       Path(".logs"),     # Application logs
       Path(".config"),   # Configuration files
       Path("models"),    # Model storage
       Path("data"),      # Data storage
   ]
   ```

2. Cache Structure:

   ```text
   .cache/
   ├── models/           # Model cache
   │   ├── llama/       # Model-specific cache
   │   ├── gpt4/        # GPT-4 response cache
   │   └── claude/      # Claude response cache
   ├── training/        # Training cache
   │   ├── datasets/    # Dataset cache
   │   └── metrics/     # Metrics cache
   └── system/          # System cache
       ├── pytest/      # Test cache
       ├── mypy/        # Type check cache
       └── temp/        # Temporary files
   ```

3. Logs Structure:

   ```text
   .logs/
   ├── app/            # Application logs
   │   ├── error/      # Error logs
   │   ├── access/     # Access logs
   │   └── debug/      # Debug logs
   ├── models/         # Model-specific logs
   │   ├── llama/      # Llama model logs
   │   ├── gpt4/       # GPT-4 logs
   │   └── claude/     # Claude logs
   └── system/         # System logs
       ├── setup/      # Setup logs
       └── monitor/    # Monitoring logs
   ```

### Cache Management

1. Cache Configuration:

   ```python
   class CacheConfig:
       """Cache configuration."""
       
       BASE_DIR = Path(".cache")
       CACHE_DIRS = {
           "models": BASE_DIR / "models",
           "training": BASE_DIR / "training",
           "system": BASE_DIR / "system",
       }
       
       CACHE_LIMITS = {
           "models": 1024 * 1024 * 1024,    # 1GB
           "training": 512 * 1024 * 1024,    # 512MB
           "system": 256 * 1024 * 1024,      # 256MB
       }
   ```

2. Cache Manager:

   ```python
   class CacheManager:
       """Manage application cache."""
       
       def __init__(self):
           self.config = CacheConfig()
           self._setup_directories()
       
       def _setup_directories(self) -> None:
           """Create cache directories."""
           for cache_dir in self.config.CACHE_DIRS.values():
               cache_dir.mkdir(parents=True, exist_ok=True)
       
       def clean_cache(self, cache_type: str) -> None:
           """Clean specific cache type."""
           if cache_dir := self.config.CACHE_DIRS.get(cache_type):
               shutil.rmtree(cache_dir)
               cache_dir.mkdir(parents=True)
   ```

### Logging

1. Log Configuration:

   ```python
   class LogConfig:
       """Logging configuration."""
       
       BASE_DIR = Path(".logs")
       LOG_DIRS = {
           "app": BASE_DIR / "app",
           "models": BASE_DIR / "models",
           "system": BASE_DIR / "system",
       }
       
       LOG_FORMATS = {
           "app": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
           "models": "%(asctime)s - %(model)s - %(levelname)s - %(message)s",
           "system": "%(asctime)s - %(component)s - %(levelname)s - %(message)s",
       }
   ```

2. Log Manager:

   ```python
   class LogManager:
       """Manage application logging."""
       
       def __init__(self):
           self.config = LogConfig()
           self._setup_directories()
           self._configure_logging()
       
       def _setup_directories(self) -> None:
           """Create log directories."""
           for log_dir in self.config.LOG_DIRS.values():
               log_dir.mkdir(parents=True, exist_ok=True)
       
       def get_logger(
           self,
           name: str,
           log_type: str = "app"
       ) -> logging.Logger:
           """Get logger with appropriate configuration."""
           logger = logging.getLogger(name)
           
           # Configure handler
           log_file = self.config.LOG_DIRS[log_type] / f"{name}.log"
           handler = logging.FileHandler(log_file)
           handler.setFormatter(
               logging.Formatter(self.config.LOG_FORMATS[log_type])
           )
           
           logger.addHandler(handler)
           return logger
   ```

### Integration

1. System Integration:

   ```python
   class SystemManager:
       """Manage system resources."""
       
       def __init__(self):
           self.cache_manager = CacheManager()
           self.log_manager = LogManager()
       
       def initialize(self) -> None:
           """Initialize system resources."""
           self.cache_manager._setup_directories()
           self.log_manager._setup_directories()
       
       def cleanup(self) -> None:
           """Clean up system resources."""
           for cache_type in self.cache_manager.config.CACHE_DIRS:
               self.cache_manager.clean_cache(cache_type)
   ```

2. Application Integration:

   ```python
   class Application:
       """Main application class."""
       
       def __init__(self):
           self.system_manager = SystemManager()
           self.logger = self.system_manager.log_manager.get_logger(__name__)
       
       async def startup(self) -> None:
           """Initialize application."""
           try:
               self.system_manager.initialize()
               self.logger.info("Application initialized successfully")
           except Exception as e:
               self.logger.error(f"Initialization failed: {e}")
               raise
   ```

### Cleanup Management

1. Cache Cleanup:

   ```python
   def clean_cache() -> None:
       """Clean application cache."""
       cache_dir = Path(".cache")
       if cache_dir.exists():
           shutil.rmtree(cache_dir)
           cache_dir.mkdir()
   ```

2. Log Rotation:

   ```python
   def rotate_logs() -> None:
       """Rotate application logs."""
       log_dir = Path(".logs")
       for log_file in log_dir.rglob("*.log"):
           if log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
               rotate_log_file(log_file)
   ```

### Testing

1. Cache Testing:

   ```python
   @pytest.mark.utility
   class TestCacheManager:
       """Test cache management."""
       
       def test_cache_setup(self):
           """Test cache directory setup."""
           manager = CacheManager()
           manager._setup_directories()
           
           for cache_dir in manager.config.CACHE_DIRS.values():
               assert cache_dir.exists()
   ```

2. Log Testing:

   ```python
   @pytest.mark.utility
   class TestLogManager:
       """Test log management."""
       
       def test_log_setup(self):
           """Test log directory setup."""
           manager = LogManager()
           manager._setup_directories()
           
           for log_dir in manager.config.LOG_DIRS.values():
               assert log_dir.exists()
   ```
