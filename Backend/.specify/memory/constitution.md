<!--
Sync Impact Report:
- Version change: 0.1.0 → 1.0.0
- Modified principles: All principles replaced with RAG chatbot-specific content
- Added sections: Accuracy and Faithfulness, Code Quality, RAG Best Practices, Security and Efficiency, User-Centric Design, Scalability
- Removed sections: None (completely new content)
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/templates/commands/*.md ⚠ pending (no command files found)
- Follow-up TODOs: None
-->

# Integrated RAG Chatbot for Published Book Content Constitution

## Core Principles

### Accuracy and Faithfulness to Book Content
All responses must be grounded in retrieved or selected text from the book content. No hallucinations or fabricated information are allowed. The system must always reference the source material when providing answers. This ensures trustworthiness and reliability for users seeking accurate information from published content.

### Excellent Code Quality
All code must be clean, modular, well-commented, and type-hinted Python code. This includes proper documentation, adherence to PEP 8 standards, comprehensive error handling, and maintainable architecture. Code reviews must verify these standards before merging.

### RAG System Best Practices
The system must implement proper chunking strategies, metadata handling, and hybrid search capabilities when needed. Proper indexing, retrieval quality, and citation mechanisms must be implemented to ensure effective knowledge extraction from book content. Semantic chunking and overlap strategies should be employed where appropriate.

### Security and Efficiency
All API keys and sensitive information must be stored in environment variables. The system must implement async operations where possible, include rate limiting mechanisms, and optimize resource usage. Data privacy and secure handling of user queries must be prioritized.

### User-Centric Design
The system must seamlessly support both full-book queries and user-selected text queries. The interface should be intuitive and responsive, providing clear feedback to users. The experience should be consistent across different query types and handle edge cases gracefully.

### Scalability and Cost-Effectiveness
The system must leverage Qdrant Cloud Free Tier and Neon free limits efficiently. Architecture decisions should consider long-term scalability while staying within free tier constraints. Resource usage should be optimized to minimize costs as usage grows.

## Technology Standards

### API Usage Requirements
The system must use Cohere API exclusively for embeddings (embed-english-v3.0 or latest multilingual) and chat (Command R or R+ with built-in RAG connectors or manual document passing). No OpenAI usage is permitted. All Cohere features must be properly configured for optimal performance.

### Data Storage and Management
Vector store: Qdrant Cloud (free tier) with proper collection setup and metadata for source tracking (e.g., page/chapter). Database: Neon Serverless Postgres for storing chat history, user sessions, or book metadata if needed (using async SQLAlchemy or psycopg). Proper indexing and query optimization must be implemented.

### Backend Architecture
Backend must be built with FastAPI featuring async endpoints, Pydantic models, proper error handling, and comprehensive OpenAPI documentation. The system must include proper request validation, response formatting, and monitoring capabilities.

### Ingestion Pipeline Requirements
The system must include a script to load book content (PDF/text), chunk intelligently (semantic or fixed-size with overlap), embed with Cohere, and upsert to Qdrant. The pipeline must handle various file formats and maintain metadata integrity throughout the process.

### RAG Implementation Modes
The system must support both full retrieval from Qdrant and direct grounding on user-selected text. When using user-selected text, it must be passed as documents to Cohere Chat. Both modes must provide consistent quality and citation capabilities.

### Citation and Source Tracking
Cohere's built-in citations must be enabled for all responses to provide traceability to book sources. Citations must be properly formatted and clearly indicate the source of information. The system must maintain accurate metadata linking responses to specific book sections.

## Development Workflow

### Project Structure Requirements
The project must follow a clear folder structure: ingestion/, api/, db/, frontend/, tests/. Dependencies must be minimal and properly pinned (FastAPI, uvicorn, cohere, qdrant-client, langchain optional for helpers, psycopg or asyncpg for Neon). All dependencies must be documented in requirements files.

### Testing Standards
Unit and integration tests must be included for ingestion, retrieval, and chat endpoints. Test coverage must be comprehensive and include edge cases, error conditions, and performance scenarios. All features must have corresponding tests before deployment.

### Frontend Implementation
The frontend must be either a simple Streamlit or Gradio interface, or pure HTML/JS widget embeddable in a published book (e.g., via iframe or script tag). The interface must be responsive, accessible, and provide a seamless user experience across different devices and browsers.

### Deployment and Distribution
The system must be deployable via platforms like Render, Vercel, or local deployment. The deployment process must be documented and include environment setup, dependency installation, and configuration steps. The system must be embeddable in static book pages with HTML/JS demo implementation.

## Governance

All implementation decisions must align with these constitutional principles. Any deviation from these principles requires explicit approval and documentation of the trade-offs. Code reviews must verify compliance with all principles, particularly accuracy, security, and code quality standards. Changes to this constitution require project stakeholder approval and must include a migration plan for existing implementations.

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17
