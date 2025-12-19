# Research: Backend-Only Focused Development for Integrated RAG Chatbot

**Date**: 2025-12-17
**Feature**: Backend-Only Focused Development for Integrated RAG Chatbot
**Spec**: /mnt/d/ai-book/backend/specs/001-rag-chatbot/spec.md

## Overview

Research summary for implementing a backend-only RAG chatbot for published book content with two query modes (full-book retrieval and selected-text grounding).

## Technology Decisions

### 1. Python 3.11+ for Backend Development
**Decision**: Use Python 3.11+ as specified in requirements
**Rationale**: Required by project constraints, excellent ecosystem for AI/ML and web development
**Alternatives considered**:
- Python 3.10: Slightly older, missing some performance improvements
- Python 3.12: Newer, potentially less stable library support

### 2. FastAPI for Web Framework
**Decision**: Use FastAPI for the backend API
**Rationale**:
- High performance async framework
- Built-in support for async/await
- Automatic OpenAPI documentation
- Pydantic integration for request/response validation
- Excellent for API development

### 3. Cohere API for LLM and Embeddings
**Decision**: Use Cohere exclusively (no OpenAI as per constraints)
**Rationale**:
- Required by project constraints (no OpenAI)
- embed-english-v3.0 for embeddings
- command-r-plus for chat with citation support
- Built-in citation capabilities

### 4. Qdrant Cloud for Vector Storage
**Decision**: Use Qdrant Cloud with free tier
**Rationale**:
- 1GB free tier sufficient for single book
- Good Python client library
- Cosine similarity metric for embeddings
- Cloud-hosted for reliability

### 5. Neon Serverless Postgres for SQL Storage
**Decision**: Use Neon Serverless Postgres
**Rationale**:
- Free tier with compute limits
- Serverless scaling
- PostgreSQL compatibility
- Python psycopg[binary] driver support

### 6. PyMuPDF for PDF Processing
**Decision**: Use PyMuPDF (fitz) for PDF text extraction
**Rationale**:
- Excellent PDF text extraction capabilities
- Preserves page numbers
- Good performance
- Better than PyPDF2/PyPDF2ium

## Architecture Patterns

### 1. Service Layer Pattern
**Decision**: Implement service layer for business logic
**Rationale**: Separates business logic from API and database layers, making code more maintainable and testable.

### 2. Repository/CRUD Pattern
**Decision**: Use CRUD operations in db/crud.py
**Rationale**: Standard pattern for database operations, keeps database logic centralized and reusable.

### 3. Async Architecture
**Decision**: Use async/await throughout the application
**Rationale**:
- Better performance for I/O-bound operations (API calls, DB queries)
- Better resource utilization
- Required by project constraints

## Implementation Approaches

### 1. Book Chunking Strategy
**Decision**: Recursive character text splitting with 600-800 char chunks and 200 char overlap
**Rationale**:
- Balances context preservation with retrieval efficiency
- 200 char overlap helps maintain context across splits
- Good for citation accuracy

### 2. Citation Handling
**Decision**: Use Cohere's built-in citation feature with strong preamble
**Rationale**:
- Cohere has native citation support
- Strong preamble ensures accurate sourcing: "Answer only based on provided documents. If unsure, say 'I don't know'. Always cite sources."

### 3. Session Management
**Decision**: Use UUID-based session IDs stored in Neon Postgres
**Rationale**:
- Persistent conversation history
- Cross-session continuity
- Simple implementation with UUID primary keys

## Risk Mitigation

### 1. Rate Limiting
**Approach**: Implement in-memory rate limiting for free tier compliance
**Rationale**: Prevents exceeding Cohere/DB usage limits on free tier

### 2. Error Handling
**Approach**: Comprehensive error handling with graceful fallbacks
**Rationale**: Ensures system remains operational during external service issues

### 3. Resumable Ingestion
**Approach**: Make ingestion script idempotent with progress tracking
**Rationale**: Allows resuming from failures without reprocessing entire books

## External Service Integration

### 1. Cohere Integration
- Use embed-english-v3.0 for 1024-dimension embeddings
- Use command-r-plus with citation_mode="accurate"
- Batch embedding with 50-100 chunk batches for efficiency

### 2. Qdrant Integration
- Create collection with 1024-dimension vectors and cosine similarity
- Store rich metadata (page_number, chunk_index, source_file, text_length)
- Use batch upsert for efficiency

### 3. Neon Postgres Integration
- Use SQLAlchemy async for database operations
- Connection pooling for performance
- Alembic for database migrations