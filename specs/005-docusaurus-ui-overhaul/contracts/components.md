# Component Interface Contracts: Docusaurus UI Overhaul

## Overview
This document defines the interface contracts for the custom UI components in the Physical AI Textbook Docusaurus site.

## Theme Provider Contract

### Interface
```typescript
interface ThemeContextType {
  currentTheme: 'light' | 'dark';
  toggleTheme: () => void;
  themeVariables: ThemeVariables;
}

interface ThemeVariables {
  gradientPrimary: string;
  glowPrimary: string;
  colors: ColorPalette;
  typography: TypographyConfig;
}

interface ColorPalette {
  primary: string;
  secondary: string;
  textPrimary: string;
  textSecondary: string;
  bgPrimary: string;
  bgSecondary: string;
}
```

### Responsibilities
- Provide theme context to all components
- Handle theme switching between light/dark modes
- Expose theme variables for consistent styling

### Guarantees
- Theme state persists across page navigation
- All components update immediately when theme changes
- Default theme respects user's system preference

## Navbar Component Contract

### Interface
```typescript
interface NavbarProps {
  logo?: LogoConfig;
  links: NavItem[];
  showDarkModeToggle?: boolean;
  showGithubLink?: boolean;
  transparentOnHero?: boolean;
}

interface NavItem {
  label: string;
  href: string;
  target?: '_blank' | '_self';
}

interface LogoConfig {
  src: string;
  alt: string;
  href: string;
}
```

### Responsibilities
- Display navigation links consistently
- Handle theme-aware styling
- Provide access to dark mode toggle
- Maintain responsive behavior

### Guarantees
- Always visible at top of viewport
- Responsive design works on all screen sizes
- Maintains accessibility standards

## Homepage Hero Component Contract

### Interface
```typescript
interface HomepageHeroProps {
  title: string;
  subtitle?: string;
  ctas?: CTAConfig[];
  gradientBackground?: boolean;
  glowEffect?: boolean;
}

interface CTAConfig {
  label: string;
  href: string;
  variant?: 'primary' | 'secondary';
  target?: '_blank' | '_self';
}
```

### Responsibilities
- Display prominent hero section
- Apply gradient background and glow effects
- Present clear call-to-action buttons
- Maintain visual consistency with theme

### Guarantees
- Full viewport height on initial load
- Properly scaled elements on all devices
- Accessible text contrast in all themes

## Book Page Layout Contract

### Interface
```typescript
interface BookPageLayoutProps {
  children: React.ReactNode;
  tocItems?: TOCItem[];
  showSidebar?: boolean;
  sidebarWidth?: string;
}

interface TOCItem {
  id: string;
  title: string;
  level?: number;
  href: string;
  children?: TOCItem[];
}
```

### Responsibilities
- Provide consistent layout for book content
- Display and manage table of contents sidebar
- Maintain responsive design
- Handle active section highlighting

### Guarantees
- Content remains readable with proper typography
- Sidebar stays visible during navigation
- Layout adapts to different screen sizes

## Chatbot Widget Contract

### Interface
```typescript
interface ChatbotWidgetProps {
  initialMessage?: string;
  themeColor?: string;
  position?: 'bottom-right' | 'bottom-left';
}

interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatbotContextType {
  messages: ChatMessage[];
  sendMessage: (text: string) => void;
  isOpen: boolean;
  toggleChat: () => void;
  clearChat: () => void;
}
```

### Responsibilities
- Provide floating chat interface
- Handle mock conversation flow
- Maintain position in bottom corner
- Apply theme-consistent styling

### Guarantees
- Always accessible to users
- Non-intrusive to page content
- Consistent with overall design theme
- Properly positioned across screen sizes

## Footer Component Contract

### Interface
```typescript
interface FooterProps {
  columns: FooterColumn[];
  socialLinks?: SocialLink[];
  copyrightText: string;
  showLogo?: boolean;
}

interface FooterColumn {
  title: string;
  links: FooterLink[];
}

interface FooterLink {
  label: string;
  href: string;
  target?: '_blank' | '_self';
}

interface SocialLink {
  name: string;
  href: string;
  icon?: string;
}
```

### Responsibilities
- Display professional footer with organized links
- Show social media connections
- Present copyright information
- Maintain consistent styling

### Guarantees
- Always visible at bottom of page
- Responsive layout for all screen sizes
- Accessible link targets