# Data Model: Backend-Only Focused Development for Integrated RAG Chatbot

**Date**: 2025-12-17
**Feature**: Backend-Only Focused Development for Integrated RAG Chatbot
**Spec**: /mnt/d/ai-book/backend/specs/001-rag-chatbot/spec.md

## Overview

Data models for the backend RAG chatbot system including database schemas and API request/response models.

## Database Models

### 1. ChatSession
**Purpose**: Represents a user's conversation session with the chatbot

**Fields**:
- `id`: UUID (Primary Key) - Unique identifier for the session
- `created_at`: DateTime (Default: now) - Timestamp when session was created
- `updated_at`: DateTime (Default: now, On update: now) - Timestamp when session was last updated
- `user_id`: String (Optional) - Optional identifier for user (for future auth)

**Relationships**:
- One-to-Many: ChatSession → Message (via session_id foreign key)

### 2. Message
**Purpose**: Represents a single message in a conversation

**Fields**:
- `id`: UUID (Primary Key) - Unique identifier for the message
- `session_id`: UUID (Foreign Key) - References ChatSession.id
- `role`: String (Enum: "user", "assistant") - The role of the message sender
- `content`: Text - The content of the message
- `timestamp`: DateTime (Default: now) - When the message was created
- `citations`: JSON (Optional) - JSON array of citation objects with source information
- `message_type`: String (Enum: "query", "response", "system") - Type of message

**Relationships**:
- Many-to-One: Message → ChatSession (via session_id foreign key)

## API Request/Response Models (Pydantic)

### 1. ChatFullRequest
**Purpose**: Request model for full-book query mode

**Fields**:
- `query`: str - The user's question about the book content
- `session_id`: Optional[str] - Existing session ID (if continuing conversation)
- `temperature`: Optional[float] - Model temperature (default: 0.1 for accuracy)
- `max_tokens`: Optional[int] - Maximum tokens in response (default: 1000)

### 2. ChatSelectedRequest
**Purpose**: Request model for selected-text query mode

**Fields**:
- `query`: str - The user's question about the selected text
- `selected_text`: str - The text selected by the user to ground the response
- `session_id`: Optional[str] - Existing session ID (if continuing conversation)
- `temperature`: Optional[float] - Model temperature (default: 0.1 for accuracy)
- `max_tokens`: Optional[int] - Maximum tokens in response (default: 1000)

### 3. ChatResponse
**Purpose**: Response model for both query modes

**Fields**:
- `message`: str - The assistant's response to the query
- `citations`: List[dict] - Array of citation objects with source information
- `session_id`: str - The session ID (newly created if not provided)
- `response_id`: str - Unique identifier for this response

### 4. Citation
**Purpose**: Individual citation object within responses

**Fields**:
- `text`: str - The text that was cited
- `source`: str - Source identifier (e.g., "page_15", "chapter_2", "chunk_id")
- `relevance_score`: Optional[float] - Relevance score if available

### 5. HealthResponse
**Purpose**: Response model for health check endpoint

**Fields**:
- `status`: str - "healthy" if system is operational
- `timestamp`: str - ISO format timestamp of check
- `services`: dict - Status of external services (Cohere, Qdrant, Neon)

### 6. SessionHistoryResponse
**Purpose**: Response model for retrieving session history

**Fields**:
- `session_id`: str - The session identifier
- `messages`: List[Message] - List of messages in the session
- `created_at`: str - When the session was created

## Vector Database Schema (Qdrant)

### 1. BookChunk Collection
**Purpose**: Vector storage for book content chunks with metadata

**Vector Configuration**:
- `vector_size`: 1024 (for Cohere embed-english-v3.0)
- `distance`: Cosine (for semantic similarity)

**Payload Fields**:
- `text`: str - The chunked text content
- `page_number`: int - Original page number in the book
- `chunk_index`: int - Sequential index of this chunk
- `source_file`: str - Original filename
- `text_length`: int - Length of the text chunk
- `section_title`: Optional[str] - Title of the section/chapter
- `book_title`: str - Title of the book
- `created_at`: str - Timestamp when chunk was created

## Validation Rules

### ChatSession Validation
- `user_id` must be a valid string if provided (max 255 chars)
- Session must be updated when new messages are added

### Message Validation
- `role` must be one of ["user", "assistant"]
- `content` must not be empty (min 1 character)
- `citations` must be valid JSON if provided
- Message must reference an existing session

### API Request Validation
- `query` must be 1-1000 characters
- `selected_text` (for selected mode) must be 1-10000 characters
- `temperature` must be between 0.0 and 1.0
- `max_tokens` must be between 1 and 2000

### BookChunk Validation
- `text` must be 100-1000 characters (for proper chunking)
- `page_number` must be positive integer
- `chunk_index` must be non-negative integer
- `text_length` must match actual text length