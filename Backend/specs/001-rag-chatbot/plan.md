# Implementation Plan: Backend-Only Focused Development for Integrated RAG Chatbot

**Branch**: `001-rag-chatbot-backend` | **Date**: 2025-12-17 | **Spec**: /mnt/d/ai-book/backend/specs/001-rag-chatbot/spec.md

**Input**: Feature specification from `/mnt/d/ai-book/backend/specs/001-rag-chatbot/spec.md`

## Summary

Backend-only implementation of a RAG chatbot for published book content with two query modes (full-book retrieval and selected-text grounding). The backend will include ingestion pipeline, FastAPI endpoints, Cohere integration, Qdrant vector storage, and Neon Postgres for chat history. Focus on accurate citations, zero hallucinations, and embeddable frontend compatibility.

## Technical Context

**Language/Version**: Python 3.11+ (required by project constraints)
**Primary Dependencies**: FastAPI, Cohere, Qdrant-client, psycopg[binary], SQLAlchemy, PyMuPDF, python-dotenv, Pydantic
**Storage**: Neon Serverless Postgres (SQL) + Qdrant Cloud (vector)
**Testing**: pytest with mocked external services (Cohere, Qdrant, DB)
**Target Platform**: Linux server (deployment-ready for Render/Fly.io/Vercel)
**Project Type**: Web application (backend API server)
**Performance Goals**: <5s response time for 95% of requests, handle concurrent users within free tier limits
**Constraints**: Must stay within free tiers (Qdrant 1GB, Neon compute limits, Cohere trial credits), no OpenAI usage
**Scale/Scope**: Single book support, multiple concurrent users, persistent chat sessions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Accuracy and Faithfulness: Implementation ensures zero hallucinations with proper citation support
- Code Quality: Implementation includes type hints, clean architecture, and comprehensive documentation
- RAG Best Practices: Proper chunking, metadata handling, and retrieval quality mechanisms planned
- Security and Efficiency: API keys stored in env vars, async operations implemented, rate limiting included
- User-Centric Design: Both full-book and user-selected text query modes supported
- Scalability: Architecture leverages free tiers efficiently with cost optimization strategies

## Project Structure

### Documentation (this feature)
```text
backend/specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)
```text
backend/
├── ingestion/
│   └── ingest_book.py          # Book ingestion pipeline script
├── api/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic request/response models
│   └── routes/
│       ├── chat.py             # Chat endpoints
│       └── health.py           # Health check endpoints
├── db/
│   ├── models.py               # SQLAlchemy database models
│   ├── database.py             # Database connection setup
│   └── crud.py                 # Database operations
├── services/
│   ├── rag_service.py          # RAG logic and Cohere integration
│   └── qdrant_service.py       # Qdrant vector database operations
├── scripts/
│   └── init_db.py              # Database initialization script
├── tests/
│   ├── test_ingestion.py       # Ingestion pipeline tests
│   ├── test_db.py              # Database operations tests
│   ├── test_api.py             # API endpoints tests
│   └── conftest.py             # Test configuration
├── requirements.txt            # Project dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── alembic/                    # Database migration files
    ├── env.py
    ├── script.py.mako
    └── versions/
```

**Structure Decision**: Backend-focused web application structure with clear separation of concerns between ingestion, API, database, and services layers. This structure supports the backend-only development approach while maintaining clean architecture for future frontend integration.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [None] | [No violations identified] | [All constitutional principles satisfied] |
