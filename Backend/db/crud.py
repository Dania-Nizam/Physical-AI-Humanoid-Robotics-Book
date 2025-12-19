from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, func
from .models import ChatSession, Message
from typing import Optional, List
import uuid

async def create_session(db: AsyncSession, user_id: Optional[str] = None) -> ChatSession:
    """
    Create a new chat session
    """
    # Validate user_id format if provided
    if user_id is not None:
        if not user_id.strip() or len(user_id) > 100 or not user_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Invalid user_id format. Must be alphanumeric with hyphens/underscores only, max 100 characters.")

    session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=user_id
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: str) -> Optional[ChatSession]:
    """
    Get a chat session by ID
    """
    # Validate session_id format
    if not session_id.strip() or len(session_id) > 100 or not session_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError("Invalid session_id format. Must be alphanumeric with hyphens/underscores only, max 100 characters.")

    result = await db.execute(select(ChatSession).filter(ChatSession.id == session_id))
    return result.scalar_one_or_none()


async def create_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    citations: Optional[list] = None,
    message_type: str = "query"
) -> Message:
    """
    Add a message to a session
    """
    # Validate inputs
    if not session_id.strip() or len(session_id) > 100 or not session_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError("Invalid session_id format. Must be alphanumeric with hyphens/underscores only, max 100 characters.")

    if role not in ["user", "assistant"]:
        raise ValueError("Role must be either 'user' or 'assistant'")

    if not content or not content.strip():
        raise ValueError("Content cannot be empty or just whitespace")

    if len(content) > 50000:  # Reasonable limit for message content
        raise ValueError("Content too long. Maximum 50000 characters allowed.")

    if message_type not in ["query", "response", "system"]:
        raise ValueError("Message type must be 'query', 'response', or 'system'")

    message = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        citations=citations,
        message_type=message_type
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def get_session_history(db: AsyncSession, session_id: str) -> List[Message]:
    """
    Get all messages in a session, ordered by timestamp
    """
    # Validate session_id format
    if not session_id.strip() or len(session_id) > 100 or not session_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError("Invalid session_id format. Must be alphanumeric with hyphens/underscores only, max 100 characters.")

    result = await db.execute(
        select(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.timestamp)
    )
    return result.scalars().all()


async def get_recent_sessions(db: AsyncSession, user_id: Optional[str] = None, limit: int = 10) -> List[ChatSession]:
    """
    Get recent sessions for a user (or all sessions if user_id not provided)
    """
    # Validate limit
    if limit <= 0 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    # Validate user_id format if provided
    if user_id is not None:
        if not user_id.strip() or len(user_id) > 100 or not user_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Invalid user_id format. Must be alphanumeric with hyphens/underscores only, max 100 characters.")

    query = select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)

    if user_id:
        query = query.filter(ChatSession.user_id == user_id)

    result = await db.execute(query)
    return result.scalars().all()


async def update_session_timestamp(db: AsyncSession, session_id: str):
    """
    Update the updated_at timestamp for a session
    """
    # Validate session_id format
    if not session_id.strip() or len(session_id) > 100 or not session_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError("Invalid session_id format. Must be alphanumeric with hyphens/underscores only, max 100 characters.")

    session = await get_session(db, session_id)
    if session:
        session.updated_at = func.now()
        await db.commit()