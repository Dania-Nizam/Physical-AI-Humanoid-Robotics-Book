from fastapi import APIRouter, Depends, HTTPException, status, Request, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging
import re

from api.models import (
    ChatFullRequest, ChatSelectedRequest, CreateSessionRequest,
    ChatResponse, SessionHistoryResponse, SessionResponse
)
from services.rag_service import get_rag_service, RAGService
from db.database import get_db_session
from db import crud
from utils.logging_config import get_logger, log_error_with_context
from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize rate limiter for this module
limiter = Limiter(key_func=get_remote_address)

# Set up logging
logger = get_logger(__name__)

router = APIRouter()

@router.post("/full", response_model=ChatResponse)
@limiter.limit("20/minute")  # 20 requests per minute per IP for full chat
async def chat_full(
    request: Request,
    chat_request: ChatFullRequest,
    db: AsyncSession = Depends(get_db_session),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Handle full-book chat queries with citations.

    This endpoint accepts a query about the book content and returns a response
    with citations to the source material.
    """
    logger.info(f"Processing full-book chat request for session: {chat_request.session_id}")

    try:
        # Get or create session
        session_id = chat_request.session_id
        if not session_id:
            # Create a new session
            session = await crud.create_session(db)
            session_id = session.id
        else:
            # Retrieve existing session
            session = await crud.get_session(db, session_id)
            if not session:
                # If session doesn't exist, create a new one with the provided ID
                session = await crud.create_session(db, user_id=None)
                session_id = session.id  # Use the generated ID instead of the provided one for consistency
                logger.info(f"Session {chat_request.session_id} not found, created new session {session_id}")

        # Query the RAG service with proper error handling
        try:
            result = rag_service.query_full_book(
                query=chat_request.query,
                top_k=6,  # Default to 6 top results
                temperature=chat_request.temperature or 0.1,
                max_tokens=chat_request.max_tokens or 1000
            )
        except ValueError as e:
            logger.warning(f"Validation error in RAG service: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )

        # Store user query in database
        await crud.create_message(
            db=db,
            session_id=session_id,
            role="user",
            content=chat_request.query
        )

        # Store assistant response in database
        await crud.create_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=result["message"],
            citations=[c.dict() for c in result["citations"]]  # Convert citations to dict for storage
        )

        # Prepare response
        response = ChatResponse(
            message=result["message"],
            citations=result["citations"],
            session_id=session_id,
            response_id=result["response_id"]
        )

        logger.info(f"Successfully processed chat request for session {session_id}")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        log_error_with_context(
            logger_instance=logger,
            error_msg=str(e),
            context={
                "session_id": chat_request.session_id,
                "query_length": len(chat_request.query) if chat_request.query else 0,
                "endpoint": "/chat/full"
            },
            error_type="chat_processing_error"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post("/selected", response_model=ChatResponse)
@limiter.limit("20/minute")  # 20 requests per minute per IP for selected chat
async def chat_selected(
    request: Request,
    chat_request: ChatSelectedRequest,
    db: AsyncSession = Depends(get_db_session),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Handle queries based on selected text only.

    This endpoint accepts a query and selected text, and returns a response
    based only on that selected text, not the full book content.
    """
    logger.info(f"Processing selected-text chat request for session: {chat_request.session_id}")

    try:
        # Get or create session
        session_id = chat_request.session_id
        if not session_id:
            # Create a new session
            session = await crud.create_session(db)
            session_id = session.id
        else:
            # Retrieve existing session
            session = await crud.get_session(db, session_id)
            if not session:
                # If session doesn't exist, create a new one
                session = await crud.create_session(db)
                session_id = session.id

        # Query the RAG service with selected text
        try:
            result = rag_service.query_selected_text(
                query=chat_request.query,
                selected_text=chat_request.selected_text,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens
            )
        except ValueError as e:
            logger.warning(f"Validation error in RAG service: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )

        # Store user query in database
        await crud.create_message(
            db=db,
            session_id=session_id,
            role="user",
            content=chat_request.query
        )

        # Store assistant response in database
        await crud.create_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=result["message"],
            citations=[c.dict() for c in result["citations"]]  # Convert citations to dict for storage
        )

        # Prepare response
        response = ChatResponse(
            message=result["message"],
            citations=result["citations"],
            session_id=session_id,
            response_id=result["response_id"]
        )

        logger.info(f"Successfully processed selected-text chat request for session {session_id}")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing selected-text chat request: {e}")
        log_error_with_context(
            logger_instance=logger,
            error_msg=str(e),
            context={
                "session_id": chat_request.session_id,
                "query_length": len(chat_request.query) if chat_request.query else 0,
                "selected_text_length": len(chat_request.selected_text) if chat_request.selected_text else 0,
                "endpoint": "/chat/selected"
            },
            error_type="selected_chat_processing_error"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing selected-text chat request: {str(e)}"
        )


@router.post("/sessions", response_model=SessionResponse)
@limiter.limit("100/minute")  # 100 requests per minute per IP for session creation
async def create_session(
    request: Request,
    session_request: CreateSessionRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Create a new chat session
    """
    # Validate the request model which already has validation constraints
    session = await crud.create_session(db, session_request.user_id)
    return SessionResponse(session_id=session.id)


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
@limiter.limit("50/minute")  # 50 requests per minute per IP for session history
async def get_session_history(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get the conversation history for a specific session
    """
    # Validate session_id format
    if not re.match(r'^[a-zA-Z0-9-_]+$', session_id) or len(session_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id format. Must be alphanumeric with hyphens/underscores only, max 100 characters."
        )

    session = await crud.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await crud.get_session_history(db, session_id)

    # Convert to response format
    message_responses = []
    for msg in messages:
        message_responses.append({
            "id": msg.id,
            "session_id": msg.session_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "citations": msg.citations,
            "message_type": msg.message_type
        })

    return SessionHistoryResponse(
        session_id=session_id,
        messages=message_responses,
        created_at=session.created_at
    )