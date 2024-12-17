"""Logging management utilities."""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, ClassVar, Type

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Professional theme for console output
PROFESSIONAL_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow", 
        "error": "red bold",
        "section": "blue bold",
        "success": "green",
    }
)

console = Console(theme=PROFESSIONAL_THEME)


class LogTemplates:
    """Log message templates."""

    REQUEST_RECEIVED = "{source} - Received request {request_id}"
    REQUEST_COMPLETED = "{source} - Completed request {request_id} in {duration:.2f}s"
    REQUEST_FAILED = "{source} - Request {request_id} failed: {error}"
    MODEL_LOADED = "Model {model_name} loaded successfully"
    MODEL_UNLOADED = "Model {model_name} unloaded"
    MODEL_ERROR = "Error with model {model_name}: {error}"
    SYSTEM_STARTUP = "System starting up - Version {version}"
    SYSTEM_SHUTDOWN = "System shutting down"
    SYSTEM_ERROR = "System error: {error}"
    SYSTEM_WARNING = "System warning: {warning}"
    SYSTEM_INFO = "System info: {info}"
    SYSTEM_SUCCESS = "System success: {success}"


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

    SUBDIRS = {
        "app": ["error", "access", "debug"],
        "models": ["llama", "gpt4", "claude"],
        "system": ["setup", "monitor"],
    }

    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_LOG_COUNT = 5  # Number of backup files to keep


class Singleton(type):
    """Metaclass for creating singleton classes."""
    
    _instances: Dict[Type, object] = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LogManager(metaclass=Singleton):
    """Manage application logging."""

    _initialized: bool = False

    def __init__(self, template: str, config: Optional[LogConfig] = None):
        """Initialize log manager.

        Args:
            template: Log message template to use
            config: Optional log configuration
        """
        if not LogManager._initialized:
            self.template = template
            self.config = config or LogConfig()
            self.console = Console(theme=PROFESSIONAL_THEME)
            self._setup_directories()
            self._configure_logging()
            self._loggers: Dict[str, logging.Logger] = {}
            LogManager._initialized = True

    def _setup_directories(self) -> None:
        """Create log directories."""
        try:
            created_dirs = False
            # Create base directories
            for log_dir in self.config.LOG_DIRS.values():
                if not log_dir.exists():
                    log_dir.mkdir(parents=True, exist_ok=True)
                    created_dirs = True

            # Create subdirectories
            for log_type, subdirs in self.config.SUBDIRS.items():
                log_dir = self.config.LOG_DIRS[log_type]
                for subdir in subdirs:
                    subdir_path = log_dir / subdir
                    if not subdir_path.exists():
                        subdir_path.mkdir(parents=True, exist_ok=True)
                        created_dirs = True

            if created_dirs:
                console.print("[green]Log directories created successfully[/green]")
        except Exception as e:
            console.print(f"[red]Failed to create log directories: {e}[/red]")
            raise

    def _configure_logging(self) -> None:
        """Configure logging system."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Add console handler with rich formatting
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=False,  # We'll add this in the format if needed
            show_path=False,  # We'll add this in the format if needed
            console=self.console,
            log_time_format="[%X]"
        )
        
        # Use a simpler format that's easier to read
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Suppress specific loggers that are too verbose
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("accelerate").setLevel(logging.WARNING)
        logging.getLogger("bitsandbytes").setLevel(logging.WARNING)

    def get_logger(
        self, name: str, log_type: str = "app", subdir: Optional[str] = None
    ) -> logging.Logger:
        """Get logger with appropriate configuration.

        Args:
            name: Logger name
            log_type: Type of logs (app/models/system)
            subdir: Optional subdirectory for logs

        Returns:
            Configured logger instance
        """
        logger_key = f"{log_type}:{name}"
        if logger_key in self._loggers:
            return self._loggers[logger_key]

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Determine log file path
        log_dir = self.config.LOG_DIRS[log_type]
        if subdir:
            log_dir = log_dir / subdir
        log_file = log_dir / f"{name}.log"

        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(self.config.LOG_FORMATS[log_type]))
        logger.addHandler(file_handler)

        self._loggers[logger_key] = logger
        return logger

    def rotate_logs(self) -> None:
        """Rotate log files that exceed size limit."""
        for log_dir in self.config.LOG_DIRS.values():
            try:
                for log_file in log_dir.rglob("*.log"):
                    if log_file.stat().st_size > self.config.MAX_LOG_SIZE:
                        self._rotate_log_file(log_file)
            except Exception as e:
                console.print(f"[red]Failed to rotate logs: {e}[/red]")
                raise

    def _rotate_log_file(self, log_file: Path) -> None:
        """Rotate a specific log file.

        Args:
            log_file: Path to log file
        """
        try:
            # Remove oldest backup if exists
            oldest_backup = log_file.with_suffix(f".log.{self.config.MAX_LOG_COUNT}")
            if oldest_backup.exists():
                oldest_backup.unlink()

            # Rotate existing backups
            for i in range(self.config.MAX_LOG_COUNT - 1, 0, -1):
                backup = log_file.with_suffix(f".log.{i}")
                if backup.exists():
                    new_backup = log_file.with_suffix(f".log.{i + 1}")
                    backup.rename(new_backup)

            # Rename current log file
            backup_1 = log_file.with_suffix(".log.1")
            shutil.copy2(log_file, backup_1)
            log_file.write_text("")  # Clear current log file

            console.print(f"[green]Rotated log file: {log_file}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to rotate log file {log_file}: {e}[/red]")
            raise

    def archive_old_logs(self, days: int = 30) -> None:
        """Archive logs older than specified days.

        Args:
            days: Number of days before archiving
        """
        import time

        current_time = time.time()
        archive_dir = self.config.BASE_DIR / "archive"
        archive_dir.mkdir(exist_ok=True)

        try:
            for log_dir in self.config.LOG_DIRS.values():
                for log_file in log_dir.rglob("*.log*"):
                    age = current_time - log_file.stat().st_mtime
                    if age > days * 86400:  # Convert days to seconds
                        # Create archive path with timestamp
                        timestamp = datetime.fromtimestamp(log_file.stat().st_mtime).strftime(
                            "%Y%m%d"
                        )
                        archive_path = archive_dir / f"{timestamp}_{log_file.name}"

                        # Move to archive
                        shutil.move(str(log_file), str(archive_path))
                        console.print(
                            f"[green]Archived log file: {log_file} -> {archive_path}[/green]"
                        )
        except Exception as e:
            console.print(f"[red]Failed to archive logs: {e}[/red]")
            raise

    def cleanup_logs(self) -> None:
        """Clean up log files and archives."""
        try:
            # Remove all log files
            for log_dir in self.config.LOG_DIRS.values():
                if log_dir.exists():
                    shutil.rmtree(log_dir)
                log_dir.mkdir(parents=True)

            # Remove archive
            archive_dir = self.config.BASE_DIR / "archive"
            if archive_dir.exists():
                shutil.rmtree(archive_dir)

            # Recreate directory structure
            self._setup_directories()
            console.print("[green]Log cleanup completed successfully[/green]")
        except Exception as e:
            console.print(f"[red]Failed to clean up logs: {e}[/red]")
            raise
