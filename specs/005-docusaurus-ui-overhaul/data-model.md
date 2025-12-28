# Data Model: Docusaurus UI Overhaul

## Overview
This document describes the data structures and components for the UI overhaul of the Physical AI Textbook Docusaurus site. Since this is primarily a frontend UI redesign, the "data model" focuses on component structures and UI state.

## Component Models

### 1. Theme Configuration
**Purpose**: Centralized theme configuration for consistent styling

**Fields**:
- primaryGradient: string (CSS gradient definition)
- glowEffects: object (glow effect definitions)
- colors: object (color palette)
- typography: object (font definitions)
- breakpoints: object (responsive breakpoints)

**Validation**:
- All color values must be valid CSS color definitions
- Gradient definitions must follow CSS gradient syntax

### 2. Navbar Component
**Purpose**: Navigation bar with gradient accents and theme support

**Fields**:
- logo: object (logo configuration)
- links: array (navigation links)
- darkModeToggle: boolean (show/hide toggle)
- githubLink: string (GitHub repository link)
- transparentBackground: boolean (for homepage hero)

**State**:
- isScrolled: boolean (for background changes on scroll)
- isDarkMode: boolean (current theme state)

### 3. Homepage Hero Component
**Purpose**: Full-screen hero section with gradient background and glowing text

**Fields**:
- backgroundImage: string (gradient or image URL)
- title: string (main title text)
- subtitle: string (optional subtitle)
- ctas: array (call-to-action buttons)
- glowIntensity: number (glow effect strength)

**State**:
- animationComplete: boolean (for entrance animations)

### 4. Book Page Layout Component
**Purpose**: Layout for book content pages with sidebar TOC

**Fields**:
- sidebarWidth: string (width of sidebar)
- contentWidth: string (width of main content)
- stickySidebar: boolean (whether sidebar stays fixed)
- tocItems: array (table of contents items)

**State**:
- sidebarOpen: boolean (mobile sidebar visibility)
- activeSection: string (currently visible section)

### 5. Footer Component
**Purpose**: Professional footer with multi-column layout

**Fields**:
- columns: array (footer column configuration)
- socialLinks: array (social media links)
- copyrightText: string (copyright information)
- logo: object (footer logo)

### 6. Chatbot Widget Component
**Purpose**: Floating mock chatbot widget

**Fields**:
- position: object (position coordinates)
- themeColor: string (primary theme color)
- initialMessage: string (first message to show)

**State**:
- isOpen: boolean (chatbot window open/closed)
- messages: array (chat history)
- isTyping: boolean (typing indicator)

## UI State Management

### Theme Context
**Purpose**: Global theme state management for light/dark mode and color consistency

**Structure**:
- currentTheme: "light" | "dark"
- themeConfig: object (theme configuration)
- toggleTheme: function (theme switching function)

### Responsive State
**Purpose**: Manage responsive behavior across different screen sizes

**Structure**:
- currentBreakpoint: string (current responsive breakpoint)
- isMobile: boolean
- isTablet: boolean
- isDesktop: boolean

## Validation Rules

### Accessibility Compliance
- All color combinations must meet WCAG 2.1 AA contrast requirements
- All interactive elements must be keyboard accessible
- All components must support screen readers

### Performance Requirements
- CSS bundle size must remain under 200KB
- Components must render efficiently
- Animations must maintain 60fps

### Cross-browser Compatibility
- All CSS features must be supported by modern browsers
- Fallbacks must be provided for advanced CSS features