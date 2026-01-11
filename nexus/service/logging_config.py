"""
NEXUS Centralized Logging Configuration
=========================================

Provides standardized logging across all NEXUS modules.

Features:
- Structured JSON logging option
- Log level from environment
- Consistent formatting
- Shared logger factory
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_entry["extra"] = record.extra

        return json.dumps(log_entry, default=str)


class StandardFormatter(logging.Formatter):
    """Standard human-readable formatter for development."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


class LoggingConfig:
    """Centralized logging configuration."""

    # Log level from environment, default INFO
    DEFAULT_LEVEL = os.getenv("NEXUS_LOG_LEVEL", "INFO").upper()

    # Use structured JSON logging in production
    USE_STRUCTURED = os.getenv("NEXUS_LOG_STRUCTURED", "false").lower() == "true"

    # Log to file if specified
    LOG_FILE = os.getenv("NEXUS_LOG_FILE")

    _initialized: bool = False
    _root_logger: Optional[logging.Logger] = None

    @classmethod
    def initialize(cls, force: bool = False) -> None:
        """
        Initialize logging configuration.

        Args:
            force: Force re-initialization even if already done
        """
        if cls._initialized and not force:
            return

        # Get log level
        level = getattr(logging, cls.DEFAULT_LEVEL, logging.INFO)

        # Create root nexus logger
        root = logging.getLogger("nexus")
        root.setLevel(level)

        # Remove existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if cls.USE_STRUCTURED:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(StandardFormatter())

        root.addHandler(console_handler)

        # File handler if specified
        if cls.LOG_FILE:
            file_handler = logging.FileHandler(cls.LOG_FILE)
            file_handler.setLevel(level)
            file_handler.setFormatter(StructuredFormatter())
            root.addHandler(file_handler)

        # Don't propagate to root logger
        root.propagate = False

        cls._initialized = True
        cls._root_logger = root

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the nexus namespace.

        Args:
            name: Logger name (will be prefixed with 'nexus.')

        Returns:
            Configured logger
        """
        # Ensure logging is initialized
        cls.initialize()

        # Return child logger
        if name.startswith("nexus."):
            return logging.getLogger(name)
        return logging.getLogger(f"nexus.{name}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a NEXUS logger by name.

    This is the primary entry point for getting loggers.

    Args:
        name: Logger name (e.g., 'server', 'daemon', 'core')

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger("server")
        >>> logger.info("Server starting")
    """
    return LoggingConfig.get_logger(name)


def configure_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure NEXUS logging programmatically.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use JSON structured logging
        log_file: Optional file path for logging
    """
    LoggingConfig.DEFAULT_LEVEL = level.upper()
    LoggingConfig.USE_STRUCTURED = structured
    LoggingConfig.LOG_FILE = log_file
    LoggingConfig.initialize(force=True)


# Initialize on import
LoggingConfig.initialize()
