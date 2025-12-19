"""
Centralized logging configuration for the RAG Chatbot backend.

This module provides a consistent logging setup across all services
and includes structured logging capabilities.
"""

import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger
import os
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "text",  # "text" or "json"
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format of logs ("text" or "json")
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter based on format preference
    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S',
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name, using the centralized configuration.

    Args:
        name: Name of the logger (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_exception(logger_instance: logging.Logger, msg: str = "An error occurred"):
    """
    Log an exception with traceback.

    Args:
        logger_instance: Logger instance to use
        msg: Message to log with the exception
    """
    logger_instance.exception(msg)


def log_api_call(
    logger_instance: logging.Logger,
    endpoint: str,
    method: str,
    status_code: int,
    response_time: float,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Log API call with structured information.

    Args:
        logger_instance: Logger instance to use
        endpoint: API endpoint that was called
        method: HTTP method (GET, POST, etc.)
        status_code: HTTP status code returned
        response_time: Time taken to process the request in seconds
        user_id: Optional user ID
        session_id: Optional session ID
    """
    log_data = {
        "event": "api_call",
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time": response_time,
        "timestamp": datetime.utcnow().isoformat()
    }

    if user_id:
        log_data["user_id"] = user_id
    if session_id:
        log_data["session_id"] = session_id

    if status_code >= 400:
        logger_instance.warning(f"API call failed: {log_data}")
    else:
        logger_instance.info(f"API call completed: {log_data}")


def log_error_with_context(
    logger_instance: logging.Logger,
    error_msg: str,
    context: dict,
    error_type: str = "application_error"
):
    """
    Log an error with additional context information.

    Args:
        logger_instance: Logger instance to use
        error_msg: Error message
        context: Dictionary with context information
        error_type: Type of error for categorization
    """
    log_data = {
        "event": "error",
        "error_type": error_type,
        "message": error_msg,
        "context": context,
        "timestamp": datetime.utcnow().isoformat()
    }

    logger_instance.error(f"Error occurred: {log_data}")


# Initialize the logging configuration when module is imported
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" or "json"
LOG_FILE = os.getenv("LOG_FILE")  # Optional

# Set up the root logger
setup_logging(log_level=LOG_LEVEL, log_format=LOG_FORMAT, log_file=LOG_FILE)