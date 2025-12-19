from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from db.database import engine, get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv
import os
import logging
import traceback
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from typing import Callable, Any
from functools import wraps
import asyncio

# Import centralized logging and performance monitoring
from utils.logging_config import get_logger, log_api_call, log_error_with_context
from utils.performance_monitor import get_request_timeout_middleware, get_performance_metrics
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
logger.info("Rate limiter initialized")

# Performance monitoring constants
REQUEST_TIMEOUT = 30  # 30 seconds timeout for requests
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB max request size

# Import models and routes (will be created in later tasks)
from api.models import ChatFullRequest, ChatSelectedRequest, CreateSessionRequest
from api.routes import chat, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up RAG Chatbot Backend...")
    yield
    # Shutdown
    logger.info("Shutting down RAG Chatbot Backend...")

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG Chatbot for published book content",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add timeout middleware with actual timeout enforcement
app.middleware("http")(get_request_timeout_middleware(REQUEST_TIMEOUT))

# Add performance metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Endpoint to expose performance metrics."""
    return get_performance_metrics()

# Add CORS middleware (allow all for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}\nTraceback: {traceback.format_exc()}")
    try:
        log_error_with_context(
            logger_instance=logger,
            error_msg=str(exc),
            context={
                "endpoint": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("user-agent"),
                "client_host": request.client.host
            },
            error_type="unhandled_exception"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later."
            }
        )
    except Exception as log_exc:
        # If there's an error in the exception handler itself, return a basic response
        logger.error(f"Error in global exception handler: {log_exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred."
            }
        )

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.warning(f"Validation error: {exc}")
    try:
        # Get the raw errors
        raw_errors = exc.errors()

        # Sanitize errors to ensure JSON serializability
        # Remove any non-serializable objects from error contexts
        sanitized_errors = []
        for error in raw_errors:
            sanitized_error = {}
            for key, value in error.items():
                # Skip the 'ctx' field that may contain non-serializable ValueError objects
                if key == 'ctx':
                    # Create a simplified context without the original error object
                    if isinstance(value, dict) and 'error' in value:
                        sanitized_error[key] = {"error_type": type(value['error']).__name__, "message": str(value['error'])}
                    else:
                        sanitized_error[key] = value
                else:
                    sanitized_error[key] = value
            sanitized_errors.append(sanitized_error)

        details = sanitized_errors
    except Exception:
        # Fallback if there's an issue processing the validation errors
        details = [{"type": "validation_error", "msg": "Invalid request parameters"}]

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "Invalid request parameters",
            "details": details
        }
    )

# Include routes
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(health.router, tags=["health"])

# Root endpoint
@app.get("/")
@limiter.limit("100/minute")  # 100 requests per minute per IP for root endpoint
async def root(request: Request):
    logger.info("Root endpoint accessed")
    return {"message": "RAG Chatbot Backend API", "status": "running", "timestamp": datetime.now().isoformat()}

# Dependency for database session
async def get_db() -> AsyncSession:
    async for session in get_db_session():
        yield session