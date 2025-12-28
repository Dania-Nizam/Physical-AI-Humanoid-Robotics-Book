# Implementation Plan: Docusaurus UI Overhaul for Physical AI Textbook

**Branch**: `005-docusaurus-ui-overhaul` | **Date**: 2025-12-19 | **Spec**: [specs/005-docusaurus-ui-overhaul/spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-docusaurus-ui-overhaul/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Complete UI redesign for the Physical AI Textbook Docusaurus site implementing a futuristic tech/AI theme with consistent CSS variables for colors/gradients, swizzled Docusaurus components (Navbar, Footer, HomepageHeader, DocSidebar, Layout), and a custom mock chatbot component. The redesign includes a full-screen hero homepage with gradient and glowing effects, improved book pages with wide content area and sticky sidebar TOC, modern navbar with gradient accents, professional footer, and a floating mock chatbot widget - all responsive and with seamless dark/light mode support.

## Technical Context

**Language/Version**: JavaScript/TypeScript (Docusaurus v3.x), React, Node.js (v18+)
**Primary Dependencies**: Docusaurus, React, CSS Modules, Tailwind CSS (or custom CSS)
**Storage**: N/A (static site generation)
**Testing**: Jest for React components, Cypress for E2E testing
**Target Platform**: Web (static hosting on GitHub Pages)
**Project Type**: Web/frontend - Docusaurus documentation site
**Performance Goals**: <3 second page load times, 60fps animations, lightweight implementation
**Constraints**: <200KB CSS/JS bundle, WCAG 2.1 AA accessibility compliance, mobile-responsive
**Scale/Scope**: Static site for textbook content, expected <1000 pages, multiple contributors

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the project constitution, this UI redesign must:
- [SPEC-DRIVEN COMPLIANCE] Follow Spec-Kit Plus methodology with formal specifications preceding implementation (PASSED - spec.md created)
- [AI-ASSISTED IMPLEMENTATION] Leverage Claude Code and AI tools for all coding tasks (PASSED - using AI for planning and implementation)
- [REPRODUCIBILITY] Maintain complete reproducibility through version-controlled specifications, prompts, and code (PASSED - all artifacts tracked in Git)
- [TECHNOLOGY STACK COMPLIANCE] Strict adherence to predetermined technology stack: Docusaurus for static site generation (PASSED - using Docusaurus)
- [QUALITY STANDARDS] Maintain high-quality, user-focused content with type-safe, well-documented, and modular code (PASSED - following standards)
- [SECURITY-FIRST APPROACH] No exposure of API keys or sensitive information (PASSED - no sensitive info in UI overhaul)
- [PERFORMANCE STANDARDS] Ensure responsive performance and optimize for cost efficiency (PASSED - lightweight CSS/JS approach)
- [SPECIFICATION COMPLIANCE] All implementations must strictly follow the spec-driven approach (PASSED - aligning with spec.md)

## Project Structure

### Documentation (this feature)

```text
specs/005-docusaurus-ui-overhaul/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Web application
book/
├── src/
│   ├── components/
│   │   ├── Homepage/
│   │   ├── Navbar/
│   │   ├── Footer/
│   │   ├── Chatbot/
│   │   └── Layout/
│   ├── css/
│   │   └── custom.css
│   ├── pages/
│   └── theme/
│       ├── Navbar/
│       ├── Footer/
│       └── MDXComponents/
└── static/

book/docusaurus.config.js
book/sidebars.js
```

**Structure Decision**: Web application structure selected with Docusaurus-specific directories for custom components, CSS, and theme overrides. The book directory contains the Docusaurus project with custom components for the UI overhaul.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |