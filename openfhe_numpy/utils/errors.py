# ==================================================================================
#  BSD 2-Clause License
#
#  Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
#
#  All rights reserved.
#
#  Author TPOC: contact@openfhe.org
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==================================================================================
"""Logging and error handling for OpenFHE-NumPy.

This module provides:
1. A package-level logger for OpenFHE-NumPy.
2. Custom exception classes for user-facing tensor errors.
3. Convenience logging helpers.
4. Optional logging configuration through environment variables.

Logging policy
--------------
Default behavior:
    No logs are printed by OpenFHE-NumPy.

If the user sets OPENFHE_DEBUG or OPENFHE_LOG_FILE:
    OpenFHE-NumPy configures its own logging.

If the application configures logging itself:
    OpenFHE-NumPy respects that configuration.

Environment variables
---------------------
OPENFHE_DEBUG:
    Enable debug logging when set to "ON", "1", "TRUE", or "YES".

OPENFHE_LOG_FORMAT:
    Custom logging format string.

OPENFHE_LOG_FILE:
    Optional path to a log file.

OPENFHE_LOG_MAX_SIZE:
    Maximum rotating log file size in bytes. Default: 10MB.

OPENFHE_LOG_BACKUP_COUNT:
    Number of rotated backup log files. Default: 5.

Backward compatibility
----------------------
FP_ENABLE_DEBUG:
    Also enables debug logging when set to "ON", "1", "TRUE", or "YES".
"""

from __future__ import annotations

import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from typing import Optional


# =============================================================================
# Logging configuration
# =============================================================================

LOGGER_NAME = "openfhe_numpy"

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5


def _env_flag_enabled(name: str, default: str = "OFF") -> bool:
    """Return True if an environment flag is enabled."""
    return os.getenv(name, default).upper() in {"ON", "1", "TRUE", "YES"}


def debug_enabled() -> bool:
    """Return whether OpenFHE-NumPy debug logging is enabled."""
    return _env_flag_enabled("OPENFHE_DEBUG") or _env_flag_enabled("FP_ENABLE_DEBUG")


def _explicit_logging_requested() -> bool:
    """Return True if OpenFHE-NumPy should configure its own logging."""
    return (
        debug_enabled()
        or os.getenv("OPENFHE_LOG_FILE") is not None
        or os.getenv("OPENFHE_LOG_FORMAT") is not None
    )


def get_logger() -> logging.Logger:
    """Return the OpenFHE-NumPy package logger.

    By default, the logger is silent and uses ``NullHandler``. This is the
    recommended behavior for open-source libraries: importing the package should
    not unexpectedly print logs.

    If ``OPENFHE_DEBUG``, ``FP_ENABLE_DEBUG``, ``OPENFHE_LOG_FILE``, or
    ``OPENFHE_LOG_FORMAT`` is set, OpenFHE-NumPy configures its own handler.

    If an application has already configured the ``openfhe_numpy`` logger, this
    function leaves that configuration unchanged.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Respect application/test configuration.
    if logger.handlers:
        return logger

    if not _explicit_logging_requested():
        logger.addHandler(logging.NullHandler())
        logger.propagate = True
        return logger

    handler_level = logging.DEBUG if debug_enabled() else logging.INFO
    log_format = os.getenv("OPENFHE_LOG_FORMAT", _LOG_FORMAT)
    formatter = logging.Formatter(log_format)

    # Console handler is enabled only when logging is explicitly requested.
    if debug_enabled() or os.getenv("OPENFHE_LOG_FORMAT") is not None:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(handler_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Optional rotating file handler.
    log_file = os.getenv("OPENFHE_LOG_FILE")
    if log_file:
        try:
            max_file_size = int(os.getenv("OPENFHE_LOG_MAX_SIZE", str(DEFAULT_MAX_FILE_SIZE)))
            backup_count = int(os.getenv("OPENFHE_LOG_BACKUP_COUNT", str(DEFAULT_BACKUP_COUNT)))

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
            )
            file_handler.setLevel(handler_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as exc:
            warnings.warn(
                f"Could not set up OpenFHE-NumPy file logging: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    # Keep logger permissive; handlers decide what to emit.
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate messages through the root logger once we add handlers.
    logger.propagate = False

    return logger


logger = get_logger()


# =============================================================================
# Custom exceptions
# =============================================================================


class ONPError(Exception):
    """Base class for all OpenFHE-NumPy errors."""


class ONPTypeError(ONPError, TypeError):
    """Raised when an object has an invalid type."""


class ONPDimensionError(ONPError, ValueError):
    """Raised when a tensor dimension, rank, or axis is invalid."""


class ONPValueError(ONPError, ValueError):
    """Raised when an invalid value is encountered."""

    def __init__(self, message: str = "Invalid value encountered.") -> None:
        super().__init__(message)


class ONPIncompatibleShapeError(ONPError, ValueError):
    """Raised when tensor shapes are incompatible for an operation."""

    def __init__(
        self,
        shape_a: object,
        shape_b: object,
        message: Optional[str] = None,
    ) -> None:
        if message is None:
            message = f"Incompatible shapes: {shape_a} vs {shape_b}"
        super().__init__(message)


class ONPNotImplementedError(ONPError, NotImplementedError):
    """Raised when a feature is not yet implemented."""

    def __init__(self, message: str = "This feature is not implemented.") -> None:
        super().__init__(message)


class ONPNotSupportedError(ONPError, NotImplementedError):
    """Raised when a feature is intentionally not supported."""

    def __init__(self, message: str = "This feature is not supported.") -> None:
        super().__init__(message)


# =============================================================================
# Logging helpers
# =============================================================================


def ONP_INFO(message: str) -> None:
    """Log an informational message."""
    logger.info(message, stacklevel=2)


def ONP_DEBUG(message: str) -> None:
    """Log a debug message.

    Visibility is controlled by logger and handler levels.
    """
    logger.debug(message, stacklevel=2)


def ONP_WARNING(message: str) -> None:
    """Log a warning message."""
    logger.warning(message, stacklevel=2)


def ONP_ERROR(message: str) -> None:
    """Log an error message."""
    logger.error(message, stacklevel=2)


# =============================================================================
# Testing helper
# =============================================================================


def capture_logs(level: int = logging.DEBUG) -> logging.Handler:
    """Attach an in-memory log handler for tests.

    Parameters
    ----------
    level:
        Logging level to capture.

    Returns
    -------
    logging.Handler
        A handler with a ``messages`` attribute containing formatted log records.

    Example
    -------
    >>> handler = capture_logs()
    >>> ONP_DEBUG("test message")
    >>> assert any("test message" in msg for msg in handler.messages)
    """

    class MemoryHandler(logging.Handler):
        """Simple in-memory logging handler."""

        def __init__(self) -> None:
            super().__init__(level)
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(self.format(record))

    handler = MemoryHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    test_logger = get_logger()

    for existing_handler in list(test_logger.handlers):
        if isinstance(existing_handler, logging.NullHandler):
            test_logger.removeHandler(existing_handler)

    test_logger.addHandler(handler)

    # Make sure debug messages can reach the test handler.
    test_logger.setLevel(logging.DEBUG)

    return handler
