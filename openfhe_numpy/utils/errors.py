"""Logging and error handling for OpenFHE-NumPy.

This module provides:
1. A configured logger with consistent log formatting
2. Custom exceptions with stack trace information
3. Convenience functions for logging at different levels

Usage:
    from ..utils.log import ONP_DEBUG, ONP_ERROR, ONP_WARNING

    ONP_DEBUG("Processing data...")
    ONP_WARNING("Unusual input detected")
    if problem:
        ONP_ERROR("Invalid input data")
"""

import inspect
import os
import logging
import threading
import traceback
from typing import Any, Dict
from logging.handlers import RotatingFileHandler

# === Configuration ===

# Default configuration values
DEFAULT_CONFIG = {
    "enable_debug": False,
    "log_format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_file": None,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}


def get_config() -> Dict[str, Any]:
    """Get logging configuration from environment variables with defaults."""
    return {
        "enable_debug": os.getenv("OPENFHE_DEBUG", "OFF").upper() in ("ON", "1", "TRUE"),
        "log_format": os.getenv("OPENFHE_LOG_FORMAT", DEFAULT_CONFIG["log_format"]),
        "log_file": os.getenv("OPENFHE_LOG_FILE", DEFAULT_CONFIG["log_file"]),
        "max_file_size": int(os.getenv("OPENFHE_LOG_MAX_SIZE", str(DEFAULT_CONFIG["max_file_size"]))),
        "backup_count": int(os.getenv("OPENFHE_LOG_BACKUP_COUNT", str(DEFAULT_CONFIG["backup_count"]))),
    }


# Load configuration (done once at module import time)
_config = get_config()

# For backward compatibility
ENABLE_DEBUG = _config["enable_debug"]
if os.getenv("FP_ENABLE_DEBUG", "OFF").upper() == "ON":
    ENABLE_DEBUG = True
    _config["enable_debug"] = True

_logger = None
_logger_lock = threading.Lock()


def get_logger() -> logging.Logger:
    """Thread-safe singleton logger with console + optional rotating file output."""
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = logging.getLogger("openfhe_numpy")

                if not _logger.handlers:
                    formatter = logging.Formatter(_config["log_format"])

                    # Rotating file handler (optional)
                    log_file = _config["log_file"]
                    if log_file:
                        try:
                            file_handler = RotatingFileHandler(
                                log_file,
                                maxBytes=_config["max_file_size"],
                                backupCount=_config["backup_count"],
                            )
                            file_handler.setFormatter(formatter)
                            _logger.addHandler(file_handler)
                        except Exception as e:
                            print(f"Warning: Could not set up file logging: {e}")

                    # Console handler
                    stream_handler = logging.StreamHandler()
                    stream_handler.setFormatter(formatter)
                    stream_handler.setLevel(logging.DEBUG if _config["enable_debug"] else logging.INFO)
                    _logger.addHandler(stream_handler)

                    _logger.setLevel(logging.DEBUG)
                    _logger.propagate = False
    return _logger


# === Custom Exceptions ===


class ONPError(Exception):
    """Base class for OpenFHE-NumPy errors."""

    def __init__(self, message: str):
        stack = traceback.extract_stack(limit=2)[-2]
        filename = os.path.basename(stack.filename)
        function_name = stack.name
        line_number = stack.lineno
        full_message = f'{message}\n    File: "{filename}", line {line_number}, in {function_name}'
        super().__init__(full_message)


class ONPTypeError(ONPError):
    """Raised when incorrect types are provided."""

    pass


class ONPDimensionError(ONPError):
    """Raised when an invalid axis is provided."""

    pass


class ONPValueError(ONPError):
    """Raised when an invalid value is encountered."""

    def __init__(self, message: str = "Invalid value encountered."):
        super().__init__(message)


class ONPImcompatibleShape(ONPValueError):
    """Raised when tensor shapes are incompatible."""

    def __init__(self, shape_a, shape_b, message: str = None):
        if message is None:
            message = f"Incompatible shapes: {shape_a} vs {shape_b}"
        super().__init__(message)


class ONPNotImplementedError(ONPError):
    """Raised when a feature is not implemented."""

    def __init__(self, message: str = "This feature is not implemented."):
        super().__init__(f"{message}")


# === Logging Helpers ===


def _format_log(level: str, message: str, stack_level: int = 2) -> str:
    frame = inspect.stack()[stack_level]
    filename = os.path.basename(frame.filename)
    function_name = frame.function
    line_number = frame.lineno
    return f'[{level}] {message}\n    File: "{filename}", line {line_number}, in {function_name}'


def _log(level: str, message: str, stack_level: int = 2) -> None:
    formatted_message = _format_log(level, message, stack_level)
    logger = get_logger()
    if level == "ONP_ERROR":
        logger.error(formatted_message)
    elif level == "ONP_DEBUG" and ENABLE_DEBUG:
        logger.debug(formatted_message)
    elif level == "ONP_WARNING":
        logger.warning(formatted_message)
    else:
        return


# === Public API ===


def ONP_INFO(message: str) -> None:
    _log("ONP_INFO", message)


def ONP_ERROR(message: str, raise_exception: bool = True) -> None:
    _log("ONP_ERROR", message)
    if raise_exception:
        raise ONPError(message)


def ONP_DEBUG(message: str) -> None:
    _log("ONP_DEBUG", message)


def ONP_WARNING(message: str) -> None:
    _log("ONP_WARNING", message)


def capture_logs(level: int = logging.DEBUG) -> logging.Handler:
    """Capture logs for testing.

    Returns a handler that collects logs for inspection in tests.

    Parameters
    ----------
    level : int, optional
        Logging level to capture (default: logging.DEBUG)

    Returns
    -------
    logging.Handler
        A handler with a 'messages' attribute containing captured log messages

    Example
    -------
    handler = capture_logs()
    ONP_DEBUG("Test message")
    assert "Test message" in handler.messages
    """

    class MemoryHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level)
            self.messages = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(self.format(record))

    handler = MemoryHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    get_logger().addHandler(handler)
    return handler
