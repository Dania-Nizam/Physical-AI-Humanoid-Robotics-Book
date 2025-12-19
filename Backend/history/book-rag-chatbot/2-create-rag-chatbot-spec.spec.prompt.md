---
id: 2
title: Create RAG Chatbot Spec
stage: spec
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: book-rag-chatbot
branch: 001-book-rag-chatbot
user: user
command: /sp.specify
labels: [spec,rag,chatbot,book-content,cohere,qdrant]
links:
  spec: /mnt/d/ai-book/specs/001-book-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-book-rag-chatbot/spec.md
 - specs/001-book-rag-chatbot/checklists/requirements.md
tests:
---

## Prompt

```
Integrated RAG Chatbot for Published Book Content using Cohere, Qdrant, Neon, and FastAPI
Target outcome: A fully functional, deployable, and embeddable Retrieval-Augmented Generation (RAG) chatbot that answers questions accurately based on a published book's content, with two modes:
1. Full-book queries (retrieves relevant chunks from the entire book via Qdrant)
2. Selected-text queries (grounds answers only on user-highlighted/selected text, no retrieval)

Core functionality to deliver:
- Book ingestion pipeline: Load book (PDF or plain text), intelligent chunking (semantic or recursive with overlap), generate Cohere embeddings, store in Qdrant with metadata (page/chapter/section for citations)
- FastAPI backend with async endpoints for:
  - Chat (full-book mode with retrieval + Cohere RAG)
  - Chat on selected text (pass user-provided text directly as documents to Cohere)
  - Health check and basic session management
- Neon Serverless Postgres integration: Store chat history, user sessions, and optional book metadata
- Simple embeddable frontend (HTML + JS widget or Streamlit/Gradio demo) that can be placed inside a published digital book (e.g., via iframe or script tag)
- Proper citation support using Cohere's built-in citation feature (show source chunks/pages)

Success criteria:
- Chatbot answers book-related questions accurately with zero hallucinations (always grounded)
- Provides traceable citations (page/section references) for full-book queries
- Correctly handles selected-text mode without accessing full knowledge base
- Ingestion script successfully processes a sample book and populates Qdrant
- Backend runs locally and can be deployed (e.g., Render, Fly.io, Vercel)
- Frontend widget is lightweight and embeddable in static HTML/book page
- All secrets handled via environment variables
- Comprehensive README with setup, ingestion, running, and embedding instructions
- Includes basic tests (ingestion, retrieval, chat endpoint)

Technical specifications & credentials (use exactly these):
- Cohere API Key: gnLhIBig2A6Mji66PEBLx70KzsnzmxDQUJF8rps6
- Qdrant Cloud Endpoint: https://33a17435-7fad-4712-92b7-d5a6acd8f994.europe-west3-0.gcp.cloud.qdrant.io
- Qdrant API Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Iz_ZITr9qgGiL6cjbLR9bTK61Z7Xs9WamHkQ5umPILw
- Qdrant Cluster ID: 33a17435-7fad-4712-92b7-d5a6acd8f994
- Neon Postgres Connection URL: postgresql://neondb_owner:npg_gICNdrwY5pO7@ep-still-meadow-a4lcind5-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
- Use Cohere models: embed-english-v3.0 (or latest v3) for embeddings, command-r-plus (or latest Command R+) for chat with citation support

Constraints:
- Use ONLY Cohere (no OpenAI, no Anthropic, no other LLMs)
- Stay within free tiers (Qdrant 1GB cluster, Neon free compute, Cohere trial/free credits)
- Project must be in Python 3.11+
- Use FastAPI + Uvicorn, async where possible
- Database migrations with Alembic or simple SQL scripts
- No LangChain (optional only if it significantly simplifies; prefer direct Cohere/Qdrant clients)
- Clean project structure: folders like ingestion/, api/, db/, frontend/, tests/, docs/

Not building:
- User authentication system (public chatbot is fine)
- Multi-book support (single book only)
- Advanced UI/UX (keep frontend minimal and embeddable)
- Real-time collaboration or complex session persistence beyond basic history
- Mobile app or native desktop client
- Paid-tier features or scaling beyond free limits

Deliverables:
- Complete codebase with .env.example
- Ingestion script (e.g., python ingest_book.py --file book.pdf)
- Running FastAPI server
- Embeddable chat widget demo
- Detailed README.md and setup guide
```

## Response snapshot

Created comprehensive feature specification for the Integrated RAG Chatbot for Published Book Content with three user stories, functional requirements, key entities, and success criteria. Created branch 001-book-rag-chatbot with complete spec file and validation checklist.

## Outcome

- ✅ Impact: Complete feature specification created with all required sections
- 🧪 Tests: N/A
- 📁 Files: Created spec.md and requirements checklist in specs/001-book-rag-chatbot/
- 🔁 Next prompts: Ready for /sp.plan to create implementation plan
- 🧠 Reflection: Specification addresses all core requirements with clear user scenarios and measurable outcomes

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin with /sp.plan to create implementation plan