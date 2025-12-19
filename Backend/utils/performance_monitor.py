"""
Performance monitoring and timeout utilities for the RAG Chatbot backend.

This module provides utilities for monitoring performance metrics,
enforcing request timeouts, and collecting performance data.
"""

import time
import asyncio
from typing import Callable, Any, Dict, Optional
from functools import wraps
import logging
from contextlib import contextmanager
import psutil
import os
from datetime import datetime
from utils.logging_config import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Class to monitor and track performance metrics."""

    def __init__(self):
        self.metrics = {
            'request_count': 0,
            'error_count': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'max_response_time': 0.0,
        }
        self.start_time = time.time()

    def record_request(self, response_time: float, is_error: bool = False):
        """Record a request's performance metrics."""
        self.metrics['request_count'] += 1
        if is_error:
            self.metrics['error_count'] += 1
        self.metrics['total_response_time'] += response_time
        self.metrics['avg_response_time'] = (
            self.metrics['total_response_time'] / self.metrics['request_count']
        )
        if response_time > self.metrics['max_response_time']:
            self.metrics['max_response_time'] = response_time

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Add system metrics
        process = psutil.Process(os.getpid())
        system_metrics = {
            'uptime_seconds': uptime,
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
        }

        return {**self.metrics, **system_metrics}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def timeout_handler(timeout_seconds: int = 30):
    """Decorator to enforce function execution timeout."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # For async functions, use asyncio.wait_for
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't enforce timeout as easily
            # Just log execution time and warn if it exceeds threshold
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            if execution_time > timeout_seconds:
                logger.warning(f"Function {func.__name__} took {execution_time:.2f}s (threshold: {timeout_seconds}s)")

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def monitor_performance(operation_name: str = "operation"):
    """Context manager to monitor execution time of a code block."""
    start_time = time.time()
    error_occurred = False

    try:
        yield
    except Exception:
        error_occurred = True
        raise
    finally:
        execution_time = time.time() - start_time
        performance_monitor.record_request(execution_time, error_occurred)

        logger.info(
            f"Performance: {operation_name} took {execution_time:.3f}s "
            f"(error: {error_occurred})"
        )


def get_request_timeout_middleware(timeout_seconds: int = 30):
    """Factory function to create a timeout middleware."""
    async def timeout_middleware(request, call_next):
        """Middleware that enforces request timeout."""
        request_start_time = time.time()

        try:
            # Use asyncio.wait_for to enforce timeout on the entire request
            response = await asyncio.wait_for(
                call_next(request),
                timeout=timeout_seconds
            )

            # Calculate response time and record metrics
            response_time = time.time() - request_start_time
            performance_monitor.record_request(response_time)

            # Add response time header for monitoring
            response.headers["X-Response-Time"] = str(response_time)
            response.headers["X-Request-Id"] = getattr(request.state, 'request_id', 'unknown')

            return response

        except asyncio.TimeoutError:
            # Log timeout
            response_time = time.time() - request_start_time
            performance_monitor.record_request(response_time, is_error=True)

            from fastapi.responses import JSONResponse
            from fastapi import status

            logger.warning(f"Request timed out after {timeout_seconds}s: {request.url.path}")

            return JSONResponse(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                content={
                    "error": "Request timeout",
                    "message": f"Request took longer than {timeout_seconds} seconds to complete"
                }
            )
        except Exception as e:
            # Record error in metrics
            response_time = time.time() - request_start_time
            performance_monitor.record_request(response_time, is_error=True)

            # Re-raise the exception to be handled by other middleware
            raise

    return timeout_middleware


def get_performance_metrics():
    """Get current performance metrics."""
    return performance_monitor.get_metrics()