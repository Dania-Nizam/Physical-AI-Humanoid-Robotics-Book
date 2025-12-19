"""
Health check API routes.

This module defines the health check endpoint for the application.
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging
import cohere
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models import HealthResponse
from db.database import get_db_session
from services.qdrant_service import get_qdrant_service
from utils.logging_config import get_logger, log_error_with_context

# Initialize rate limiter for this module
limiter = Limiter(key_func=get_remote_address)

# Load environment variables
load_dotenv()

# Set up logging
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
@limiter.limit("60/minute")  # 60 requests per minute per IP for health check
async def health_check(
    request: Request,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Health check endpoint.

    This endpoint returns the health status of the application and its services.
    """
    logger.info("Health check requested")

    # Initialize services status
    services_status = {
        "database": "disconnected",
        "qdrant": "disconnected",
        "cohere": "disconnected"
    }

    # Check database connectivity
    try:
        # Test database connection by attempting a simple query
        await db.execute("SELECT 1")
        services_status["database"] = "connected"
        logger.info("Database connectivity check passed")
    except Exception as e:
        logger.error(f"Database connectivity check failed: {e}")
        log_error_with_context(
            logger_instance=logger,
            error_msg=str(e),
            context={"service": "database", "check_type": "connectivity"},
            error_type="health_check_error"
        )
        services_status["database"] = "disconnected"

    # Check Qdrant connectivity
    try:
        qdrant_service = get_qdrant_service()
        # Test Qdrant connection by getting collection info
        collection_info = qdrant_service.get_collection_info()
        services_status["qdrant"] = "connected"
        logger.info(f"Qdrant connectivity check passed, collection info: {collection_info}")
    except Exception as e:
        logger.error(f"Qdrant connectivity check failed: {e}")
        log_error_with_context(
            logger_instance=logger,
            error_msg=str(e),
            context={"service": "qdrant", "check_type": "connectivity"},
            error_type="health_check_error"
        )
        services_status["qdrant"] = "disconnected"

    # Check Cohere connectivity
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not set")

        cohere_client = cohere.Client(api_key=cohere_api_key)
        # Test Cohere connection with a simple model check or embed call
        test_embedding = cohere_client.embed(
            texts=["health check"],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        services_status["cohere"] = "connected"
        logger.info("Cohere connectivity check passed")
    except Exception as e:
        logger.error(f"Cohere connectivity check failed: {e}")
        log_error_with_context(
            logger_instance=logger,
            error_msg=str(e),
            context={"service": "cohere", "check_type": "connectivity"},
            error_type="health_check_error"
        )
        services_status["cohere"] = "disconnected"

    # Determine overall health status
    all_connected = all(status == "connected" for status in services_status.values())
    overall_status = "healthy" if all_connected else "unhealthy"

    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services=services_status
    )

    logger.info(f"Health check completed with status: {overall_status}")
    return response