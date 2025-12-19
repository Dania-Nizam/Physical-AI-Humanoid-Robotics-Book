---
id: 3
title: Create RAG Chatbot Backend Plan
stage: plan
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: rag-chatbot
branch: 001-rag-chatbot-backend
user: user
command: /sp.plan
labels: [plan,rag,chatbot,backend,cohere,qdrant,neon]
links:
  spec: /mnt/d/ai-book/backend/specs/001-rag-chatbot/spec.md
  plan: /mnt/d/ai-book/backend/specs/001-rag-chatbot/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - backend/specs/001-rag-chatbot/plan.md
 - backend/specs/001-rag-chatbot/research.md
 - backend/specs/001-rag-chatbot/data-model.md
 - backend/specs/001-rag-chatbot/quickstart.md
 - backend/specs/001-rag-chatbot/contracts/api-contract.yaml
tests:
---

## Prompt

```
Updated: Backend-Only Focused Development for Integrated RAG Chatbot (Frontend Deferred)

Phase 1: Project Setup and Foundation (1 day)
- Create complete project structure with folders: ingestion/, api/, db/, tests/, docs/, scripts/
- Initialize .gitignore (exclude .env, __pycache__, etc.)
- Create .env.example with all required variables:
  COHERE_API_KEY=gnLhIBig2A6Mji66PEBLx70KzsnzmxDQUJF8rps6
  QDRANT_URL=https://33a17435-7fad-4712-92b7-d5a6acd8f994.europe-west3-0.gcp.cloud.qdrant.io
  QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Iz_ZITr9qgGiL6cjbLR9bTK61Z7Xs9WamHkQ5umPILw
  QDRANT_COLLECTION_NAME=book_chunks
  NEON_DATABASE_URL=postgresql://neondb_owner:npg_gICNdrwY5pO7@ep-still-meadow-a4lcind5-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
- Create requirements.txt with minimal pinned dependencies:
  fastapi>=0.115.0
  uvicorn[standard]>=0.30.0
  cohere>=5.0.0
  qdrant-client>=1.11.0
  psycopg[binary]>=3.1.0
  sqlalchemy>=2.0.0
  alembic>=1.13.0
  python-dotenv>=1.0.0
  pydantic>=2.8.0
  pymupdf>=1.24.0  # for PDF ingestion
  tqdm>=4.66.0
  pytest>=8.0.0
  httpx>=0.27.0
- Write initial README.md with:
  - Project overview
  - Backend-only focus (frontend will be added later)
  - Setup instructions
  - Ingestion and API usage

Phase 2: Book Ingestion Pipeline (2 days)
- Develop ingestion/ingest_book.py CLI script
- Accept input: --file path/to/book.pdf or .txt
- Extract text with page numbers (PyMuPDF for PDF, simple read for text)
- Intelligent chunking: recursive split, target 600-800 characters per chunk, 200 char overlap
- Add rich metadata: page_number, chunk_index, source_file, text_length
- Batch embed chunks using Cohere embed-english-v3.0 (batch size 50-100)
- Create Qdrant collection (vector size 1024 for v3, cosine metric) if not exists
- Upsert vectors with payload
- Full progress tracking, retry logic, and dry-run mode
- Make script resumable and idempotent

Phase 3: Neon Postgres Database Layer (1-2 days)
- db/models.py: Define SQLAlchemy models
  - BookChunk (optional, if storing text separately)
  - ChatSession (id, created_at, user_id optional)
  - Message (id, session_id, role, content, timestamp, citations json)
- db/database.py: Async engine setup with connection pooling
- db/crud.py: Functions for create_session, add_message, get_history
- migrations/: Use Alembic for initial schema (sessions and messages tables)
- init_db.py script to run migrations

Phase 4: FastAPI Backend Core (3-4 days)
- api/main.py: FastAPI app with lifespan for DB connection
- Add CORS middleware (allow all for now)
- Pydantic models:
  - ChatFullRequest(query: str, session_id: str | None)
  - ChatSelectedRequest(query: str, selected_text: str, session_id: str | None)
  - ChatResponse(message: str, citations: list[dict], session_id: str)
- Endpoints:
  - GET /health → return status
  - POST /chat/full → Qdrant search (top_k=6), format documents, call Cohere Chat with citation_mode="accurate"
  - POST /chat/selected → directly pass selected_text as documents to Cohere (split into chunks if too long)
  - GET /sessions/{session_id}/history → return conversation history
  - POST /sessions → create new session
- Use strong preamble in Cohere chat: "Answer only based on provided documents. If unsure, say 'I don't know'. Always cite sources."
- Store user query + assistant response + citations in DB
- Async everything possible (Qdrant, Cohere, DB)

Phase 5: Testing and Backend Validation (2 days)
- tests/:
  - test_ingestion.py (mock Cohere/Qdrant, verify chunking/metadata)
  - test_db.py (session and message CRUD)
  - test_api.py (full flow with mocked external services using responses or pytest-httpx)
- Manual testing:
  - Ingest a small sample book (e.g., public domain PDF)
  - Test both /chat/full and /chat/selected endpoints via curl or HTTP client
  - Verify citations appear and point to correct pages/chunks
  - Confirm selected-text mode ignores full knowledge base
- Add API documentation via FastAPI OpenAPI (/docs)

Phase 6: Polish and Deployment Prep
- Final logging, error handling, input validation
- Rate limiting (simple in-memory if needed)
- Update README with:
  - How to run ingestion
  - How to start server (uvicorn api.main:app --reload)
  - Example curl commands for both chat modes
  - Deployment notes (Render, Fly.io, Railway – free tiers)
- Ensure all secrets are loaded from .env only

Total Estimated Timeline: 9-12 days (backend complete and fully testable)
Final Deliverable: Robust, production-ready backend API for RAG chatbot
- Accurate full-book queries with citations
- Isolated selected-text queries
- Persistent chat history in Neon
- Ready for future frontend integration

Frontend development will be handled in a separate phase later.
```

## Response snapshot

Completed comprehensive planning for the backend RAG chatbot, including implementation plan, research, data models, quickstart guide, and API contracts. Created structured documentation in the backend/specs/001-rag-chatbot directory with all required planning artifacts.

## Outcome

- ✅ Impact: Complete backend planning completed with all required artifacts
- 🧪 Tests: N/A (planning phase)
- 📁 Files: Created plan.md, research.md, data-model.md, quickstart.md, and contracts/api-contract.yaml
- 🔁 Next prompts: Ready for /sp.tasks to generate implementation tasks
- 🧠 Reflection: Planning addresses all backend requirements with clear architecture and implementation approach

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Generate tasks with /sp.tasks command