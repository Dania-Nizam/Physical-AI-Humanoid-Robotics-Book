# Quickstart Guide: Backend-Only Focused Development for Integrated RAG Chatbot

**Date**: 2025-12-17
**Feature**: Backend-Only Focused Development for Integrated RAG Chatbot
**Spec**: /mnt/d/ai-book/backend/specs/001-rag-chatbot/spec.md

## Overview

Quick start guide to set up, run, and test the backend RAG chatbot for published book content.

## Prerequisites

- Python 3.11+
- pip package manager
- Git (optional, for cloning)
- Access to Cohere API (API key)
- Access to Qdrant Cloud
- Access to Neon Postgres

## Setup Instructions

### 1. Clone and Navigate to Project
```bash
cd /path/to/ai-book/backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:
```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=https://33a17435-7fad-4712-92b7-d5a6acd8f994.europe-west3-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=book_chunks
NEON_DATABASE_URL=postgresql://neondb_owner:npg_gICNdrwY5pO7@ep-still-meadow-a4lcind5-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

### 5. Initialize Database
```bash
python scripts/init_db.py
# Or using alembic:
alembic upgrade head
```

## Ingesting a Book

### 1. Prepare Your Book
Ensure your book is in PDF or plain text format.

### 2. Run Ingestion Script
```bash
python ingestion/ingest_book.py --file path/to/your/book.pdf
```

Optional parameters:
```bash
python ingestion/ingest_book.py --file path/to/your/book.pdf --dry-run  # Test without actually ingesting
python ingestion/ingest_book.py --file path/to/your/book.pdf --chunk-size 700 --overlap 200  # Custom chunking
```

### 3. Verify Ingestion
Check that chunks were properly stored in Qdrant with correct metadata.

## Running the Backend Server

### 1. Start the API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access API Documentation
Open your browser to `http://localhost:8000/docs` for interactive API documentation.

## Testing the API

### 1. Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
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

### 2. Full-Book Query Mode
```bash
curl -X POST http://localhost:8000/chat/full \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main theme of this book?",
    "session_id": "optional-session-id"
  }'
```

### 3. Selected-Text Query Mode
```bash
curl -X POST http://localhost:8000/chat/selected \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain this concept in simpler terms",
    "selected_text": "The complex concept is defined here in the book...",
    "session_id": "optional-session-id"
  }'
```

### 4. Get Session History
```bash
curl http://localhost:8000/sessions/{session_id}/history
```

## Example Usage Scenarios

### Scenario 1: First-Time User
1. Ingest a book: `python ingestion/ingest_book.py --file my_book.pdf`
2. Start server: `uvicorn api.main:app --reload`
3. Query the book: Use POST to `/chat/full` endpoint
4. Session automatically created with new session_id

### Scenario 2: Continuing Conversation
1. Use the same session_id in subsequent requests
2. The system will maintain conversation context
3. History is stored in Neon Postgres

### Scenario 3: Selected Text Mode
1. User selects text in the book interface
2. Send selected text with query to `/chat/selected` endpoint
3. Response will be grounded only on the provided text

## API Endpoints

### Health Check
- `GET /health` - Check system status

### Chat Operations
- `POST /chat/full` - Full-book query with retrieval
- `POST /chat/selected` - Selected-text query without retrieval
- `GET /sessions/{session_id}/history` - Get conversation history
- `POST /sessions` - Create new session

## Troubleshooting

### Common Issues

1. **Environment Variables Not Loaded**
   - Ensure `.env` file is properly configured
   - Verify virtual environment is activated

2. **Qdrant Connection Issues**
   - Check QDRANT_URL and QDRANT_API_KEY
   - Verify network connectivity to Qdrant Cloud

3. **Database Connection Issues**
   - Verify NEON_DATABASE_URL is correct
   - Check if database is properly initialized

4. **Cohere API Issues**
   - Confirm COHERE_API_KEY is valid
   - Check if you're within free tier limits

### Testing Individual Components

1. Test Cohere connection:
```python
import cohere
client = cohere.Client(os.getenv("COHERE_API_KEY"))
response = client.chat(message="Hello")
print(response)
```

2. Test Qdrant connection:
```python
from qdrant_client import QdrantClient
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
collections = client.get_collections()
print(collections)
```

3. Test database connection:
```bash
python -c "from db.database import engine; import asyncio; asyncio.run(engine.dispose())"
```

## Testing Instructions

### 1. Run Unit Tests
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=.

# Run specific test file
pytest tests/test_api.py

# Run tests with verbose output
pytest tests/ -v

# Run tests and stop on first failure
pytest tests/ -x
```

### 2. Test Individual Components
```bash
# Test API endpoints
curl -X GET http://localhost:8000/health

# Test rate limiting (make multiple requests quickly)
for i in {1..25}; do curl -X POST http://localhost:8000/chat/full \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query ' $i '"}' & done

# Test database operations
python -c "from db.database import get_db_session; from sqlalchemy import text; import asyncio; async def test(): async with get_db_session() as session: result = await session.execute(text('SELECT 1')); print(result.scalar_one())"; asyncio.run(test())

# Test Qdrant connectivity
python -c "from services.qdrant_service import get_qdrant_service; service = get_qdrant_service(); print(service.get_collection_info())"

# Test Cohere connectivity
python -c "import cohere, os; client = cohere.Client(os.getenv('COHERE_API_KEY')); response = client.embed(texts=['test'], model='embed-english-v3.0'); print(f'Embedding successful: {len(response.embeddings[0])} dimensions')"
```

### 3. Performance Testing
```bash
# Test API response times
time curl -X POST http://localhost:8000/chat/full \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this book about?"}'

# Test concurrent requests
# Using Apache Bench (install with: apt-get install apache2-utils or brew install ab)
ab -n 100 -c 10 http://localhost:8000/health
```

### 4. Integration Testing
```bash
# Test complete flow: ingest -> query -> history
1. Ingest a sample book: python ingestion/ingest_book.py --file path/to/sample.pdf
2. Query the book: curl -X POST http://localhost:8000/chat/full -H "Content-Type: application/json" -d '{"query": "Summarize this book"}'
3. Check session history: curl http://localhost:8000/sessions/{session_id}/history
```

### 5. Error Testing
```bash
# Test validation errors
curl -X POST http://localhost:8000/chat/full \
  -H "Content-Type: application/json" \
  -d '{"query": ""}'  # Should return 422 validation error

curl -X POST http://localhost:8000/chat/selected \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "selected_text": ""}'  # Should return 422 validation error

# Test rate limiting
# Make more than 20 requests per minute to /chat/full to trigger rate limit (429)
```

## Next Steps

1. **Complete Testing**: Run the full test suite with `pytest tests/` and verify all tests pass
2. **Load Testing**: Use tools like Apache Bench or Locust for load testing
3. **Production Deployment**: Deploy to Render, Fly.io, or Vercel using the deployment guides
4. **Frontend Integration**: Connect to the embeddable frontend widget
5. **Monitoring Setup**: Set up logging and monitoring for production use
6. **Security Hardening**: Implement authentication, input sanitization, and security headers for production