<!-- SYNC IMPACT REPORT
Version change: N/A -> 1.0.0
Modified principles: N/A (new constitution)
Added sections: All principles and sections added
Removed sections: N/A
Templates requiring updates:
- .specify/templates/plan-template.md ✅ updated
- .specify/templates/spec-template.md ✅ updated
- .specify/templates/tasks-template.md ✅ updated
- .specify/templates/commands/*.md ✅ reviewed
- README.md ⚠ pending
Follow-up TODOs: None
-->

# Unified AI/Spec-Driven Book with Embedded RAG Chatbot Constitution

## Core Principles

### Spec-Driven Development First
All project artifacts must follow Spec-Kit Plus methodology with formal specifications preceding implementation. Every feature, requirement, and architectural decision must be documented in spec files before coding begins. This ensures reproducible, version-controlled development with clear acceptance criteria.

### AI-Assisted Implementation
Leverage Claude Code and AI tools for all coding and content generation tasks where applicable. All code and content must be generated and tested via AI assistance to maintain consistency and accelerate development. Human oversight remains critical for quality assurance and validation.

### Reproducibility and Version Control (NON-NEGOTIABLE)
Complete reproducibility through version-controlled specifications, prompts, and code. All changes must be tracked in Git with clear commit messages. Specifications, prompt history records, and architecture decisions are first-class artifacts that must be maintained alongside code.

### Quality and User-Focused Content
Maintain high-quality, accurate, and user-focused book content with seamless integration of interactive RAG chatbot functionality. All content must undergo validation for accuracy and usability before release. Code must be type-safe, well-documented, and modular.

### Technology Stack Compliance
Strict adherence to predetermined technology stack: Docusaurus for static site generation, GitHub Pages for deployment, OpenAI Agents SDK/ChatKit for chatbot, FastAPI backend, Neon Serverless Postgres for storage, and Qdrant Cloud for vector database. Deviations require formal approval and justification.

### Security-First Approach
No exposure of API keys or sensitive information in code; use environment variables exclusively. All security practices must follow industry standards with zero tolerance for hardcoded credentials or secrets in version control.

## Additional Constraints and Standards

Technology stack requirements: Use TypeScript for type safety, Docusaurus for documentation site generation, GitHub Pages for hosting, Neon Serverless Postgres for relational data, and Qdrant Cloud Free Tier for vector storage. Optimize for free tier limitations to minimize costs. Chatbot must support both full book content queries and selected text-only context-aware retrieval.

Deployment policies: Deploy to GitHub repository with GitHub Pages for the book site. Ensure all deliverables are version-controlled with clear spec artifacts. Maintain zero critical bugs in deployment or chatbot functionality through comprehensive testing.

Performance standards: Ensure responsive chatbot performance with reasonable latency for user queries. Optimize for cost efficiency while maintaining functionality within free tier constraints.

## Development Workflow and Quality Gates

Code review requirements: All pull requests must include specification alignment verification, code quality assessment, and security scanning. Automated testing must pass before merge approval. Changes to specifications require corresponding implementation updates.

Testing gates: Manual testing for usability and accuracy is mandatory before release. All functionality must pass validation including book content accessibility, chatbot response accuracy, and context-aware retrieval features. Automated tests must cover critical paths.

Specification compliance: All implementations must strictly follow the spec-driven approach. Any deviations from the original specifications require formal documentation and approval before implementation.

## Governance

This constitution supersedes all other development practices and methodologies. All team members must comply with these principles and standards. Amendments require formal documentation, approval from project stakeholders, and a migration plan for existing codebase. All PRs and reviews must verify constitutional compliance before approval. Complexity must be justified with clear benefits outweighing the added maintenance burden.

**Version**: 1.0.0 | **Ratified**: 2025-12-16 | **Last Amended**: 2025-12-16
