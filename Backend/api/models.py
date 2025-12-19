from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

# Request models
class ChatFullRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="User query about the book content")
    session_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9-_]+$', max_length=100, description="Session ID for conversation history")
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature (0.0-2.0)")
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=4000, description="Maximum number of tokens in response (1-4000)")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        if len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return v.strip()

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9-_]+$', v):
                raise ValueError('Session ID can only contain alphanumeric characters, hyphens, and underscores')
            if len(v) > 100:
                raise ValueError('Session ID must be at most 100 characters')
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v

    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None and (v < 1 or v > 4000):
            raise ValueError('Max tokens must be between 1 and 4000')
        return v

    @model_validator(mode='after')
    def validate_model_consistency(self):
        # Additional cross-field validation
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError('Temperature must be non-negative')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('Max tokens must be at least 1')
        return self


class ChatSelectedRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="User query about the selected text")
    selected_text: str = Field(..., min_length=1, max_length=10000, description="Text selected/highlighted by user")
    session_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9-_]+$', max_length=100, description="Session ID for conversation history")
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature (0.0-2.0)")
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=4000, description="Maximum number of tokens in response (1-4000)")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        if len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return v.strip()

    @field_validator('selected_text')
    @classmethod
    def validate_selected_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Selected text cannot be empty or just whitespace')
        if len(v.strip()) < 5:
            raise ValueError('Selected text must be at least 5 characters long')
        return v.strip()

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9-_]+$', v):
                raise ValueError('Session ID can only contain alphanumeric characters, hyphens, and underscores')
            if len(v) > 100:
                raise ValueError('Session ID must be at most 100 characters')
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v

    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None and (v < 1 or v > 4000):
            raise ValueError('Max tokens must be between 1 and 4000')
        return v

    @model_validator(mode='after')
    def validate_model_consistency(self):
        # Additional cross-field validation
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError('Temperature must be non-negative')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('Max tokens must be at least 1')
        return self


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9-_]+$', max_length=100, description="User ID associated with the session")

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9-_]+$', v):
                raise ValueError('User ID can only contain alphanumeric characters, hyphens, and underscores')
            if len(v) > 100:
                raise ValueError('User ID must be at most 100 characters')
        return v


# Response models
class Citation(BaseModel):
    text: str
    source: str
    relevance_score: Optional[float] = None


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
    message_type: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageResponse]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]


class SessionResponse(BaseModel):
    session_id: str