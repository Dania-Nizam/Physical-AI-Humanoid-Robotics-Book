# RAG Chatbot Backend for Published Book Content

Backend-only implementation of a RAG (Retrieval-Augmented Generation) chatbot for published book content with two query modes:
1. Full-book queries (retrieves relevant chunks from the entire book via Qdrant)
2. Selected-text queries (grounds answers only on user-highlighted/selected text, no retrieval)

## Features

- Accurate answers with proper citations to book pages/sections
- Zero hallucinations - responses always grounded in book content
- Persistent chat sessions with history
- Support for PDF and plain text book formats
- Embeddable in static HTML/book pages

## Prerequisites

- Python 3.11+
- Access to Cohere API
- Access to Qdrant Cloud
- Access to Neon Postgres

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

   Required environment variables:
   - `COHERE_API_KEY`: Your Cohere API key
   - `QDRANT_URL`: Your Qdrant Cloud URL
   - `QDRANT_API_KEY`: Your Qdrant API key
   - `NEON_DATABASE_URL`: Your Neon Postgres database URL

4. Initialize database:
   ```bash
   python scripts/init_db.py
   # Or using alembic: alembic upgrade head
   ```

## Usage

### Ingest a Book

The ingestion script processes PDF and plain text files and stores them in Qdrant for RAG queries:

```bash
# Basic ingestion
python ingestion/ingest_book.py path/to/your/book.pdf

# With custom collection name
python ingestion/ingest_book.py path/to/your/book.pdf --collection my_book_collection

# With custom chunk size and overlap
python ingestion/ingest_book.py path/to/your/book.pdf --chunk-size 700 --overlap 200

# Dry run to see what would be ingested
python ingestion/ingest_book.py path/to/your/book.pdf --dry-run

# Resume from a previous partial ingestion
python ingestion/ingest_book.py path/to/your/book.pdf --resume
```

The ingestion process:
- Extracts text with page numbers from PDFs (or reads plain text files)
- Chunks text intelligently with overlap to preserve context
- Adds rich metadata (page numbers, chunk indices, source file, text length)
- Creates embeddings using Cohere's embed-english-v3.0 model
- Stores vectors with payloads in Qdrant collection
- Includes progress tracking, retry logic, and resumable operations

### Start the API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### API Usage Examples

#### Full-book Query (with citations)
```bash
curl -X POST "http://localhost:8000/chat/full" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main theme of this book?",
    "session_id": "optional-session-id",
    "temperature": 0.1,
    "max_tokens": 1000
  }'
```

Response:
```json
{
  "response": "The main theme of this book is...",
  "citations": [
    {
      "text": "Excerpt from the book...",
      "page_number": 42,
      "source_file": "book.pdf",
      "chunk_index": 5
    }
  ],
  "session_id": "generated-or-provided-session-id"
}
```

#### Selected-text Query
```bash
curl -X POST "http://localhost:8000/chat/selected" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this selected text mean?",
    "selected_text": "The selected text the user highlighted...",
    "session_id": "optional-session-id",
    "temperature": 0.1,
    "max_tokens": 1000
  }'
```

Response:
```json
{
  "response": "Based only on the provided text...",
  "citations": [
    {
      "text": "The selected text...",
      "source": "user_input"
    }
  ],
  "session_id": "generated-or-provided-session-id"
}
```

#### Get Session History
```bash
curl -X GET "http://localhost:8000/sessions/your-session-id/history"
```

Response:
```json
{
  "session_id": "your-session-id",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What is the main theme?",
      "timestamp": "2025-12-17T10:00:00Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "The main theme is...",
      "citations": [...],
      "timestamp": "2025-12-17T10:00:05Z"
    }
  ]
}
```

#### Create New Session
```bash
curl -X POST "http://localhost:8000/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "optional-user-id"
  }'
```

Response:
```json
{
  "session_id": "newly-generated-session-id",
  "user_id": "optional-user-id",
  "created_at": "2025-12-17T10:00:00Z"
}
```

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-17T10:00:00Z",
  "services": {
    "cohere": "connected",
    "qdrant": "connected",
    "neon": "connected"
  }
}
```

### API Documentation
Open your browser to `http://localhost:8000/docs` for interactive API documentation.

## Endpoints

- `GET /health` - Health check
- `POST /chat/full` - Full-book query mode with citations
- `POST /chat/selected` - Selected-text query mode
- `GET /sessions/{session_id}/history` - Get conversation history
- `POST /sessions` - Create new session

## Architecture

- FastAPI backend with async support
- Cohere for embeddings and chat
- Qdrant Cloud for vector storage
- Neon Postgres for session and message history
- PyMuPDF for PDF processing

## Rate Limiting

The API includes built-in rate limiting to comply with free tier limits:
- 100 requests per minute per IP address
- 10 concurrent connections max
- Request timeout: 30 seconds

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (validation errors)
- `404` - Resource not found
- `422` - Validation error (malformed request)
- `429` - Rate limit exceeded
- `500` - Internal server error

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=.
```