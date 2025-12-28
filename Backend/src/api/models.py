from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

# --- Request Models ---

class ChatFullRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 1000

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class ChatSelectedRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    selected_text: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 1000

class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None

# --- Response Models ---

class Citation(BaseModel):
    text: str
    source: str
    relevance_score: Optional[float] = 0.0  # Default value set to 0.0

class ChatResponse(BaseModel):
    message: str
    citations: List[Citation]
    session_id: str
    response_id: str

class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    citations: Optional[List[Citation]] = None
    message_type: str = "text"

class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageResponse]
    created_at: datetime

class SessionResponse(BaseModel):
    session_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]