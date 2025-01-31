"""Logging configuration."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class LogConfig:
    """Configuration for logging."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[Path] = None
    console_output: bool = True
    file_output: bool = False
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    propagate: bool = False
    handlers: List[str] = None
    filters: List[str] = None
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.handlers is None:
            self.handlers = ["console"]
        if self.filters is None:
            self.filters = []
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            "level": self.level,
            "format": self.format,
            "date_format": self.date_format,
            "file_path": str(self.file_path) if self.file_path else None,
            "console_output": self.console_output,
            "file_output": self.file_output,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
            "propagate": self.propagate,
            "handlers": self.handlers,
            "filters": self.filters
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LogConfig":
        """Create config from dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            LogConfig instance
        """
        return cls(**config) 