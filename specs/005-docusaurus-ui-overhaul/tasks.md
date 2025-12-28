---
description: "Task list for Docusaurus UI overhaul implementation"
---

# Tasks: Docusaurus UI Overhaul for Physical AI Textbook

**Input**: Design documents from `/specs/005-docusaurus-ui-overhaul/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Docusaurus project**: `book/src/`, `book/static/`, `book/pages/`, `book/components/`
- **CSS/Theme**: `book/src/css/custom.css`
- **Swizzled Components**: `book/src/theme/`
- **Configuration**: `book/docusaurus.config.js`

## Phase 1: Setup (Minimal Infrastructure)

**Purpose**: Configure CSS variables and theme system for the new UI

- [X] T001 Set up CSS variables for teal-purple gradient theme in book/src/css/custom.css
- [X] T002 Configure theme context for light/dark mode support in book/src/css/custom.css

---

## Phase 2: User Story 1 - Access Modern Homepage (Priority: P1) 🎯 MVP

**Goal**: Implement a futuristic hero homepage with gradient background, glowing title, and clear navigation options

**Independent Test**: The homepage can be viewed with a modern, visually appealing design featuring gradient backgrounds, glowing text effects, and clear navigation options

### Implementation for User Story 1

- [X] T003 [P] Create homepage hero section with gradient background in book/src/components/Homepage/Hero.js
- [X] T004 [P] Implement glowing text effects for homepage title in book/src/css/custom.css
- [X] T005 Add call-to-action buttons with futuristic styling in book/src/css/custom.css
- [X] T006 Integrate new homepage hero with existing Docusaurus layout in book/src/components/Homepage/Hero.js

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 3: User Story 2 - Navigate Through Book Content (Priority: P1)

**Goal**: Implement clean, readable book pages with sidebar table of contents and improved typography

**Independent Test**: Book pages display with clean layout, readable typography, and functional sidebar navigation that works in both light and dark modes

### Implementation for User Story 2

- [X] T007 [P] Swizzle DocSidebar component to enhance styling in book/src/theme/DocSidebar/index.js
- [X] T008 [P] Create enhanced typography styles for content readability in book/src/css/custom.css
- [X] T009 Update book page layout for improved content area in book/src/theme/Layout/index.js
- [X] T010 Add sticky sidebar behavior in book/src/theme/DocSidebar/index.js

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 4: User Story 3 - Use Consistent Navigation (Priority: P2)

**Goal**: Implement consistent navigation elements (navbar, footer) throughout the site with gradient accents

**Independent Test**: The navbar and footer appear consistently across all pages with gradient accents and proper functionality in both light and dark modes

### Implementation for User Story 3

- [X] T011 [P] Swizzle Navbar component to enhance styling in book/src/theme/Navbar/index.js
- [X] T012 [P] Swizzle Footer component with multi-column layout in book/src/theme/Footer/index.js
- [X] T013 Add theme-consistent styling to navbar in book/src/css/custom.css
- [X] T014 Add theme-consistent styling to footer in book/src/css/custom.css

**Checkpoint**: All user stories should now be independently functional

---

## Phase 5: User Story 4 - Interact with AI Chatbot (Priority: P2)

**Goal**: Implement a mock floating chatbot widget that appears consistently across all pages with matching theme

**Independent Test**: The floating chatbot widget appears in the bottom-right corner, has futuristic design matching the theme, and responds to user interactions

### Implementation for User Story 4

- [X] T015 Create floating chatbot component in book/src/components/Chatbot/ChatbotWidget.js
- [X] T016 Implement chatbot UI with glow effects in book/src/components/Chatbot/ChatbotWidget.js
- [X] T017 Add chatbot styles to match theme in book/src/css/custom.css
- [X] T018 Integrate chatbot component site-wide in book/src/theme/Layout/index.js

**Checkpoint**: All user stories should now be fully implemented and testable

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T019 [P] Implement accessibility enhancements across all UI components
- [X] T020 Performance optimization for CSS and JavaScript
- [X] T021 Cross-browser compatibility testing and fixes
- [X] T022 Mobile responsiveness testing and adjustments
- [X] T023 Final theme consistency check across all components

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **User Stories (Phase 2+)**: All depend on Setup phase completion
  - User stories can proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Setup (Phase 1) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Setup (Phase 1) - No dependencies on other stories
- **User Story 3 (P3)**: Can start after Setup (Phase 1) - No dependencies on other stories
- **User Story 4 (P4)**: Can start after Setup (Phase 1) - No dependencies on other stories

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- Once Setup phase completes, all user stories can start in parallel (if team capacity allows)
- Different user stories can be worked on in parallel by different team members

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: User Story 1
3. **STOP and VALIDATE**: Test User Story 1 independently
4. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence