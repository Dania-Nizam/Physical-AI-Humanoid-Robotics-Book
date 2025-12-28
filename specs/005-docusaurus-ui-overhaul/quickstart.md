# Quickstart Guide: Docusaurus UI Overhaul

## Overview
This guide provides developers with the essential information to work with the new futuristic UI for the Physical AI Textbook Docusaurus site.

## Project Setup

### Prerequisites
- Node.js v18+ installed
- npm or yarn package manager
- Git for version control

### Initial Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-book
   ```

2. Navigate to the book directory:
   ```bash
   cd book
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm run start
   ```

## Key Files and Directories

### CSS Customization
- `src/css/custom.css` - Main CSS file with theme variables and custom styles
- Contains CSS variables for colors, gradients, and theme settings

### Swizzled Components
- `src/theme/Navbar/` - Custom navigation component
- `src/theme/Footer/` - Custom footer component
- `src/theme/Layout/` - Custom layout wrapper
- `src/theme/DocSidebar/` - Enhanced documentation sidebar
- `src/components/Homepage/` - Homepage hero section components

### Chatbot Component
- `src/components/Chatbot/` - Mock chatbot widget implementation
- Includes React component and styling

## Theme Customization

### CSS Variables
The theme uses CSS variables for consistent styling. Key variables include:
- `--gradient-primary`: Main gradient (teal to purple)
- `--glow-primary`: Glow effect for interactive elements
- `--text-primary`: Primary text color
- `--bg-primary`: Primary background color

### Dark/Light Mode
Theme switching is handled automatically through Docusaurus' `useColorMode` hook. All components respond to theme changes.

## Component Development

### Adding New Styles
1. Define CSS variables in `src/css/custom.css`
2. Use variables in component styles for consistency
3. Ensure styles work in both light and dark modes

### Swizzling Components
To further customize a Docusaurus component:
```bash
npm run swizzle @docusaurus/theme-classic ComponentName -- --typescript --wrap
```

### Chatbot Customization
The mock chatbot is implemented as a React component. To modify:
1. Update responses in the component state
2. Adjust styling in the component CSS
3. Modify behavior in the component logic

## Build and Deployment

### Local Build
```bash
npm run build
```

### Development Build
```bash
npm run serve
```

## Common Tasks

### Updating Color Scheme
1. Modify CSS variables in `src/css/custom.css`
2. Verify all components render correctly in both themes
3. Test accessibility contrast ratios

### Adding New Pages
New pages will automatically use the custom theme components
1. Create new MDX files in the `docs/` directory
2. Add to sidebar configuration in `sidebars.js`
3. Ensure responsive design works correctly

### Testing Responsive Design
- Use browser developer tools to test different screen sizes
- Verify chatbot widget positioning across devices
- Check that gradients and effects scale properly