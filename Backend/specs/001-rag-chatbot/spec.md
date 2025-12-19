# Feature Specification: Integrated RAG Chatbot for Published Book Content

**Feature Branch**: `001-book-rag-chatbot`
**Created**: 2025-12-17
**Status**: Draft
**Input**: User description: "Integrated RAG Chatbot for Published Book Content using Cohere, Qdrant, Neon, and FastAPI
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
- Detailed README.md and setup guide"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Query Book Content with Citations (Priority: P1)

A reader wants to ask questions about a published book and receive accurate answers with citations showing the source pages/sections. The user visits the book page where the chatbot widget is embedded, types a question about the book content, and receives a response that is grounded in the book with proper citations to specific pages or sections.

**Why this priority**: This is the core functionality that delivers the main value proposition of the RAG chatbot - helping readers understand and navigate book content through AI-powered Q&A with reliable sourcing.

**Independent Test**: Can be fully tested by providing a book with known content, asking questions about that content, and verifying that responses are accurate and include proper citations to the source material.

**Acceptance Scenarios**:
1. **Given** a book has been ingested into the system, **When** a user asks a question about the book content, **Then** the system responds with an accurate answer grounded in the book and includes citations to specific pages/sections
2. **Given** a user asks a question not covered by the book content, **When** the system processes the query, **Then** the system responds that the information is not available in the book rather than hallucinating

---

### User Story 2 - Query on Selected Text (Priority: P2)

A reader has highlighted specific text in the book and wants answers grounded only on that selected text, not the entire book. The user selects text in the book, activates the chatbot in selected-text mode, asks a question related to the selection, and receives an answer based only on the selected text without accessing the full book knowledge base.

**Why this priority**: This provides an alternative interaction model that allows users to get answers specifically based on content they've identified, which is useful for detailed analysis of specific passages.

**Independent Test**: Can be tested by providing user-selected text as input and verifying that the response is based only on that text without retrieving from the broader book content.

**Acceptance Scenarios**:
1. **Given** a user has selected specific text from the book, **When** the user asks a question in selected-text mode, **Then** the system responds based only on the selected text without accessing the full book
2. **Given** user-selected text is provided, **When** the user asks a question that cannot be answered from the selection, **Then** the system indicates that the information is not available in the selected text

---

### User Story 3 - Book Content Ingestion (Priority: P3)

A content publisher or system administrator needs to process a book (PDF or plain text) and make it available for the RAG chatbot. The user runs the ingestion script with the book file as input, and the system processes the book, chunks the content intelligently, generates embeddings, and stores them in the vector database with proper metadata for citations.

**Why this priority**: This is a prerequisite for the chat functionality but can be implemented and tested independently. It enables the core functionality but doesn't directly serve end users.

**Independent Test**: Can be tested by providing a sample book file and verifying that it gets properly processed, chunked, embedded, and stored in the vector database with correct metadata.

**Acceptance Scenarios**:
1. **Given** a PDF or text file of a book, **When** the ingestion script is run, **Then** the content is properly chunked, embedded, and stored in Qdrant with metadata
2. **Given** book content has been ingested, **When** the system checks the vector database, **Then** the chunks are available with proper page/chapter/section metadata for citations

---

### Edge Cases

- What happens when a user asks a question that spans multiple book sections or chapters?
- How does the system handle very long user-selected text that might exceed API limits?
- What happens when the book content is in a format that's difficult to parse (e.g., scanned images, complex layouts)?
- How does the system handle questions that require context from multiple distant parts of the book?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST answer book-related questions with zero hallucinations (always grounded in the actual book content)
- **FR-002**: System MUST provide traceable citations (page/section references) for all answers in full-book query mode
- **FR-003**: System MUST handle selected-text queries without accessing the full knowledge base
- **FR-004**: System MUST process PDF and plain text book files through the ingestion pipeline
- **FR-005**: System MUST perform intelligent chunking of book content with appropriate overlap for context
- **FR-006**: System MUST store chat history in Neon Serverless Postgres with proper session management
- **FR-007**: System MUST support embedding the chat widget in static HTML/book pages via iframe or script tag
- **FR-008**: System MUST handle all API keys and secrets through environment variables
- **FR-009**: System MUST generate Cohere embeddings using embed-english-v3.0 model
- **FR-010**: System MUST use Cohere Command R+ model for chat with citation support enabled
- **FR-011**: System MUST store content in Qdrant Cloud with proper metadata for source tracking
- **FR-012**: System MUST provide health check endpoints for monitoring
- **FR-013**: Ingestion script MUST successfully process a sample book and populate Qdrant
- **FR-014**: System MUST run locally and support deployment to platforms like Render, Fly.io, or Vercel

### Key Entities

- **BookContent**: Represents the ingested book content, including metadata like title, author, chapters, pages, and sections. Contains the text chunks with their embeddings and source references.
- **ChatSession**: Represents a user's conversation session with the chatbot, including the history of questions and responses. Contains session-specific data and links to user interactions.
- **ChatMessage**: Represents a single message in a conversation, including the user's question, the system's response, citations, and timestamps. Links to the parent ChatSession.
- **BookChunk**: Represents a segment of the book content that has been processed for RAG, including the text, embedding vector, and source metadata (page, chapter, section).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of responses are grounded in book content with zero hallucinations when answering book-related questions
- **SC-002**: 100% of full-book query responses include accurate citations with specific page/section references
- **SC-003**: Ingestion script successfully processes sample books (PDF/text) and populates Qdrant with properly chunked content and metadata
- **SC-004**: Chatbot responds to queries within 5 seconds for 95% of requests under normal load
- **SC-005**: Selected-text mode correctly limits responses to only the provided text without accessing full knowledge base
- **SC-006**: Frontend widget loads in under 2 seconds and can be embedded in static HTML pages
- **SC-007**: All secrets are handled via environment variables with no hardcoded credentials
- **SC-008**: System passes all basic tests for ingestion, retrieval, and chat endpoints
- **SC-009**: Comprehensive README includes setup, ingestion, running, and embedding instructions
