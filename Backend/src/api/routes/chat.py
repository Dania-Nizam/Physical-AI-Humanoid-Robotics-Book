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
@limiter.limit("20/minute")
async def chat_full(
    request: Request,
    chat_request: ChatFullRequest,
    db: AsyncSession = Depends(get_db_session),
    rag_service: RAGService = Depends(get_rag_service)
):
    logger.info(f"Processing full-book chat request for session: {chat_request.session_id}")

    try:
        # 1. Session logic (Try-Except taake DB error se bot na ruke)
        session_id = chat_request.session_id or "default-session"
        try:
            if not chat_request.session_id:
                session = await crud.create_session(db)
                session_id = session.id
            else:
                session = await crud.get_session(db, chat_request.session_id)
                if not session:
                    session = await crud.create_session(db)
                    session_id = session.id
        except Exception as db_err:
            logger.warning(f"Database Session Error (Skipping DB): {db_err}")
            session_id = "temp-session"

        # 2. RAG Service Call (Asli kaam yahan ho raha hai)
        try:
            result = rag_service.query_full_book(
                query=chat_request.query,
                top_k=6,
                temperature=chat_request.temperature or 0.1,
                max_tokens=chat_request.max_tokens or 1000
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # 3. Store in DB (Isay safely try mein rakha hai)
        try:
            await crud.create_message(
                db=db, session_id=session_id, role="user", content=chat_request.query
            )
            await crud.create_message(
                db=db, session_id=session_id, role="assistant",
                content=result["message"],
                citations=[c.dict() for c in result["citations"]]
            )
        except Exception as db_err:
            logger.warning(f"Could not save message to DB: {db_err}")

        # 4. Final Response
        return ChatResponse(
            message=result["message"],
            citations=result["citations"],
            session_id=session_id,
            response_id=result["response_id"]
        )

    except Exception as e:
        logger.error(f"Main Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))