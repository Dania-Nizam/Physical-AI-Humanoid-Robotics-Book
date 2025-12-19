---
id: 1
title: Create Constitution
stage: constitution
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution,rag,chatbot,project-setup]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

```
/sp.constitution
Project: Integrated RAG Chatbot for Published Book Content
Core principles:
* Accuracy and faithfulness to book content (no hallucinations; always ground responses in retrieved or selected text)
* Excellent code quality (clean, modular, well-commented, type-hinted Python code)
* Best practices for RAG systems (proper chunking, metadata handling, hybrid search if needed, citations)
* Security and efficiency (use environment variables for keys, async where possible, rate limiting)
* User-centric design (support both full-book queries and user-selected text queries seamlessly)
* Scalability and cost-effectiveness (leverage Qdrant Cloud Free Tier and Neon free limits wisely)

Key standards:
* Use Cohere API exclusively for embeddings (embed-english-v3.0 or latest multilingual) and chat (Command R or R+ with built-in RAG connectors or manual document passing)
* Vector store: Qdrant Cloud (free tier) with proper collection setup, metadata for source tracking (e.g., page/chapter)
* Database: Neon Serverless Postgres for storing chat history, user sessions, or book metadata if needed (use async SQLAlchemy or psycopg)
* Backend: FastAPI with async endpoints, Pydantic models, proper error handling, and OpenAPI docs
* Ingestion pipeline: Script to load book (PDF/text), chunk intelligently (semantic or fixed-size with overlap), embed with Cohere, upsert to Qdrant
* RAG modes: Full retrieval from Qdrant OR direct grounding on user-selected text (pass as documents to Cohere Chat)
* Citations: Always enable Cohere's built-in citations for traceability to book sources
* Frontend embedding: Simple Streamlit or Gradio interface, or pure HTML/JS widget embeddable in a published book (e.g., via iframe or script tag)
* Testing: Include unit/integration tests for ingestion, retrieval, and chat endpoints

Constraints:
* No OpenAI usage (Cohere only)
* Stay within free tiers (Qdrant 1GB, Neon compute limits, Cohere trial/free credits)
* Project structure: Clear folders (ingestion/, api/, db/, frontend/, tests/)
* Dependencies: Minimal and pinned (FastAPI, uvicorn, cohere, qdrant-client, langchain optional for helpers, psycopg or asyncpg for Neon)
* No unnecessary features (focus on core chatbot functionality)

Success criteria:
* Chatbot accurately answers book questions with citations
* Supports user-selected text queries without full retrieval
* Fully functional ingestion script for any book text/PDF
* Deployable backend (e.g., via Render, Vercel, or local)
* Embeddable in a static book page (HTML/JS demo)
* Zero critical bugs; passes all tests
* Comprehensive README with setup, usage, and extension guides
```

## Response snapshot

Integrated RAG Chatbot for Published Book Content Constitution with comprehensive principles and standards.

## Outcome

- ✅ Impact: Created complete project constitution with 6 core principles and detailed standards
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md with project-specific principles
- 🔁 Next prompts: Proceed with project planning and implementation
- 🧠 Reflection: Constitution now provides clear guidance for RAG chatbot development

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin with project planning using /sp.plan