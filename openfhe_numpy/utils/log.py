import inspect
import os
import logging
import threading
import traceback
from logging.handlers import RotatingFileHandler

# Enable debug mode based on environment variable
FP_ENABLE_DEBUG = os.getenv("FP_ENABLE_DEBUG", "OFF").upper() == "ON"

_logger = None
_logger_lock = threading.Lock()


def get_logger():
    """Thread-safe singleton logger with console + optional rotating file output."""
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = logging.getLogger("openfhe_numpy")

                if not _logger.handlers:
                    log_format = os.getenv(
                        "OPENFHE_LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s"
                    )
                    formatter = logging.Formatter(log_format)

                    # Rotating file handler (optional)
                    log_file = os.getenv("OPENFHE_LOG_FILE")
                    if log_file:
                        try:
                            max_bytes = int(os.getenv("OPENFHE_LOG_MAX_SIZE", 10 * 1024 * 1024))
                            backup_count = int(os.getenv("OPENFHE_LOG_BACKUP_COUNT", 5))
                            file_handler = RotatingFileHandler(
                                log_file, maxBytes=max_bytes, backupCount=backup_count
                            )
                            file_handler.setFormatter(formatter)
                            _logger.addHandler(file_handler)
                        except Exception as e:
                            print(f"Warning: Could not set up file logging: {e}")

                    # Console handler
                    stream_handler = logging.StreamHandler()
                    stream_handler.setFormatter(formatter)
                    stream_handler.setLevel(logging.DEBUG if FP_ENABLE_DEBUG else logging.INFO)
                    _logger.addHandler(stream_handler)

                    _logger.setLevel(logging.DEBUG)
                    _logger.propagate = False
    return _logger


# Base Exception
class ONPError(Exception):
    """Base class for tensor-related errors."""

    def __init__(self, message: str):
        stack = traceback.extract_stack(limit=2)[-2]
        filename = os.path.basename(stack.filename)
        function_name = stack.name
        line_number = stack.lineno
        full_message = f'{message}\n    File: "{filename}", line {line_number}, in {function_name}'
        super().__init__(full_message)


# Custom Exceptions
class InvalidAxisError(ONPError):
    """Raised when an invalid axis is provided."""

    pass


class ONPNotImplementedError(ONPError):
    """Raised when a feature is not implemented."""

    def __init__(self, message: str = "This feature is not implemented."):
        super().__init__(f"{message}")


# Logging helpers
def _format_log(level: str, message: str, stack_level=2) -> str:
    frame = inspect.stack()[stack_level]
    filename = os.path.basename(frame.filename)
    function_name = frame.function
    line_number = frame.lineno
    return f'[{level}] {message}\n    File: "{filename}", line {line_number}, in {function_name}'


def _log(level: str, message: str, stack_level=2) -> None:
    formatted_message = _format_log(level, message, stack_level)
    logger = get_logger()
    if level == "ONP_ERROR":
        logger.error(formatted_message)
    elif level == "ONP_DEBUG" and FP_ENABLE_DEBUG:
        logger.debug(formatted_message)
    elif level == "ONP_WARNING":
        logger.warning(formatted_message)


# Public log APIs
def ONP_ERROR(message: str) -> None:
    _log("ONP_ERROR", message)
    raise ONPError(message)


def ONP_DEBUG(message: str) -> None:
    _log("ONP_DEBUG", message)


def ONP_WARNING(message: str) -> None:
    _log("ONP_WARNING", message)
