---
description: "Task list for backend RAG chatbot implementation"
---

# Tasks: Backend-Only Focused Development for Integrated RAG Chatbot

**Input**: Design documents from `/mnt/d/ai-book/backend/specs/001-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Basic tests requested in feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/src/` → `backend/` (all files directly in backend/)
- Paths shown below follow the plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan (ingestion/, api/, db/, services/, tests/, scripts/)
- [X] T002 Create requirements.txt with minimal pinned dependencies
- [X] T003 [P] Create .env.example with all required variables
- [X] T004 Create .gitignore (exclude .env, __pycache__, .venv/, etc.)
- [X] T005 Create initial README.md with project overview and setup instructions

---
## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create db/models.py with SQLAlchemy models (ChatSession, Message)
- [X] T007 Create db/database.py with async engine setup and connection pooling
- [X] T008 [P] Create db/crud.py with functions for create_session, add_message, get_history
- [X] T009 Create alembic/ directory with migration files (env.py, script.py.mako)
- [X] T010 Create scripts/init_db.py script to run migrations
- [X] T011 Create api/models.py with Pydantic request/response models
- [X] T012 Setup environment variable loading with python-dotenv

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---
## Phase 3: User Story 3 - Book Content Ingestion (Priority: P3) 🎯 Foundation for Core Features

**Goal**: Process a book (PDF or plain text) and make it available for the RAG chatbot

**Independent Test**: Can be fully tested by providing a sample book file and verifying that it gets properly processed, chunked, embedded, and stored in the vector database with correct metadata.

### Tests for User Story 3 (OPTIONAL - included per spec)
- [X] T013 [P] [US3] Unit test for PDF text extraction in tests/test_ingestion.py
- [X] T014 [P] [US3] Unit test for chunking logic in tests/test_ingestion.py
- [X] T015 [P] [US3] Integration test for complete ingestion flow in tests/test_ingestion.py

### Implementation for User Story 3
- [X] T016 [P] [US3] Create ingestion/ingest_book.py CLI script skeleton
- [X] T017 [US3] Implement PDF text extraction with PyMuPDF (with page numbers) in ingestion/ingest_book.py
- [X] T018 [US3] Implement plain text file reading in ingestion/ingest_book.py
- [X] T019 [US3] Implement intelligent chunking (recursive split, 600-800 chars, 200 char overlap) in ingestion/ingest_book.py
- [X] T020 [US3] Add rich metadata (page_number, chunk_index, source_file, text_length) to chunks
- [X] T021 [US3] Implement Cohere embedding using embed-english-v3.0 (batch size 50-100) in ingestion/ingest_book.py
- [X] T022 [US3] Create Qdrant collection (vector size 1024, cosine metric) if not exists
- [X] T023 [US3] Implement batch upsert of vectors with payload to Qdrant
- [X] T024 [US3] Add progress tracking, retry logic, and dry-run mode to ingestion script
- [X] T025 [US3] Make ingestion script resumable and idempotent

**Checkpoint**: At this point, User Story 3 should be fully functional and testable independently - able to ingest a book and store it in Qdrant with proper metadata

---
## Phase 4: User Story 1 - Query Book Content with Citations (Priority: P1) 🎯 MVP

**Goal**: Reader asks questions about a published book and receives accurate answers with citations showing the source pages/sections

**Independent Test**: Can be fully tested by providing a book with known content, asking questions about that content, and verifying that responses are accurate and include proper citations to the source material.

### Tests for User Story 1 (OPTIONAL - included per spec)
- [X] T026 [P] [US1] Contract test for chat/full endpoint in tests/test_api.py
- [X] T027 [P] [US1] Integration test for full-book retrieval in tests/test_api.py
- [X] T028 [P] [US1] Unit test for citation accuracy in tests/test_api.py

### Implementation for User Story 1
- [X] T029 [P] [US1] Create services/qdrant_service.py for Qdrant vector database operations
- [X] T030 [P] [US1] Create services/rag_service.py for RAG logic and Cohere integration
- [X] T031 [US1] Implement Cohere integration for chat with citation support in services/rag_service.py
- [X] T032 [US1] Implement Qdrant search (top_k=6) and document formatting in services/rag_service.py
- [X] T033 [US1] Add strong preamble for zero hallucinations in services/rag_service.py
- [X] T034 [US1] Implement POST /chat/full endpoint in api/routes/chat.py
- [X] T035 [US1] Add session management to POST /chat/full endpoint (create/retrieve session)
- [X] T036 [US1] Store user query + assistant response + citations in DB for /chat/full
- [X] T037 [US1] Implement GET /sessions/{session_id}/history endpoint in api/routes/chat.py
- [X] T038 [US1] Implement GET /health endpoint in api/routes/health.py
- [X] T039 [US1] Add CORS middleware to FastAPI app in api/main.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently - able to query full book content with citations

---
## Phase 5: User Story 2 - Query on Selected Text (Priority: P2)

**Goal**: Reader has highlighted specific text in the book and wants answers grounded only on that selected text, not the entire book

**Independent Test**: Can be tested by providing user-selected text as input and verifying that the response is based only on that text without retrieving from the broader book content.

### Tests for User Story 2 (OPTIONAL - included per spec)
- [X] T040 [P] [US2] Contract test for chat/selected endpoint in tests/test_api.py
- [X] T041 [P] [US2] Integration test for selected-text grounding in tests/test_api.py
- [X] T042 [P] [US2] Unit test for knowledge base isolation in tests/test_api.py

### Implementation for User Story 2
- [X] T043 [US2] Implement POST /chat/selected endpoint in api/routes/chat.py
- [X] T044 [US2] Implement direct pass of selected_text as documents to Cohere in rag_service.py
- [X] T045 [US2] Add logic to split long selected text into chunks if needed
- [X] T046 [US2] Ensure selected-text mode ignores full knowledge base (Qdrant search)
- [X] T047 [US2] Add session management to POST /chat/selected endpoint
- [X] T048 [US2] Store user query + selected text + assistant response + citations in DB
- [X] T049 [US2] Add validation to ensure responses are grounded only in selected text

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---
## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T050 [P] Update README.md with ingestion instructions and API usage examples
- [ ] T051 Add comprehensive error handling and logging across all services
- [ ] T052 Add input validation for all API endpoints

- [ ] T053 [P] Add additional unit tests in tests/ for edge cases
- [ ] T054 Add rate limiting (simple in-memory if needed) for free tier compliance
- [ ] T055 Update quickstart.md with complete setup and testing instructions
- [ ] T056 Create POST /sessions endpoint for explicit session creation in api/routes/chat.py
- [ ] T057 Add performance monitoring and timeout handling
- [X] T058 Run all tests and fix any issues found

---
## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in priority order (P3 → P1 → P2)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - No dependencies on other stories, but enables US1
- **User Story 1 (P1)**: Can start after Foundational (Phase 2) and requires US3 to be complete - Provides core functionality
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) and US1 - Provides alternative interaction model

### Within Each User Story

- Tests (if included) SHOULD be written and can guide implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, user stories must follow dependency order (US3 before US1)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different components within a story can be worked on in parallel if they don't depend on each other

---
## Parallel Example: User Story 1

```bash
# After foundational phase is complete:

# Launch all services for User Story 1 together:
T029 [P] [US1] Create services/qdrant_service.py for Qdrant vector database operations
T030 [P] [US1] Create services/rag_service.py for RAG logic and Cohere integration

# Launch all API development for User Story 1 together:
T034 [US1] Implement POST /chat/full endpoint in api/routes/chat.py
T037 [US1] Implement GET /sessions/{session_id}/history endpoint in api/routes/chat.py
```

---
## Implementation Strategy

### MVP First (User Story 3 + User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 3 (Ingestion - enables other stories)
4. Complete Phase 4: User Story 1 (Core chat functionality with citations)
5. **STOP and VALIDATE**: Test full-book queries with citations independently
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 3 → Ingestion capability → Test independently
3. Add User Story 1 → Full-book queries → Test independently → Deploy/Demo (MVP!)
4. Add User Story 2 → Selected-text queries → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers after foundational phase:

1. Developer A: User Story 3 (Ingestion)
2. Once US3 is complete, Developer B can work on: User Story 1 (Core chat)
3. Once US1 is complete, Developer C can work on: User Story 2 (Selected text)
4. Stories complete and integrate independently

---
## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests align with acceptance criteria from spec
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence