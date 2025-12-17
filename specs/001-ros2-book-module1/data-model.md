# Data Model: ROS 2 Book Module 1 - The Robotic Nervous System

## Content Entities

### Chapter Content
- **Name**: Chapter content for ROS 2 educational module
- **Fields**:
  - id (string): Unique identifier with sequential numbering (e.g., "01-introduction-to-ros2")
  - title (string): Chapter title
  - content (string): Markdown content (800-1500 words)
  - frontmatter (object): Docusaurus frontmatter with title, description, etc.
  - codeExamples (array): List of embedded code examples with language tags
  - diagrams (array): List of Mermaid diagrams and image references
  - wordCount (number): Actual word count (between 800-1500)
- **Relationships**: Belongs to a Module
- **Validation rules**: Must have valid Docusaurus frontmatter, word count between 800-1500, contains at least one code example

### Module Structure
- **Name**: Educational module container
- **Fields**:
  - id (string): Module identifier (e.g., "module-1-robotic-nervous-system")
  - label (string): Display label ("Module 1: The Robotic Nervous System (ROS 2)")
  - position (number): Navigation position (1 for first module)
  - chapters (array): List of Chapter Content entities
  - sidebarConfig (object): Docusaurus sidebar configuration
- **Relationships**: Contains multiple Chapter Content entities
- **Validation rules**: Must have exactly 8 chapters as specified

### Code Example
- **Name**: Educational code example for ROS 2 concepts
- **Fields**:
  - language (string): Programming language ("python" for rclpy examples)
  - code (string): Source code content
  - description (string): Explanation of the code example
  - setupInstructions (string): How to run the example
  - launchInstructions (string): Specific launch commands for ROS 2
  - dependencies (array): Required ROS 2 packages or Python libraries
- **Relationships**: Belongs to a Chapter Content
- **Validation rules**: Must be functional and tested, include clear setup instructions

### Diagram/Visualization
- **Name**: Educational diagram for ROS 2 concepts
- **Fields**:
  - type (string): "mermaid" or "image"
  - content (string): Mermaid syntax or image path
  - description (string): Explanation of what the diagram illustrates
  - caption (string): Figure caption for accessibility
- **Relationships**: Belongs to a Chapter Content
- **Validation rules**: Must clearly illustrate the concept being taught

### Docusaurus Configuration
- **Name**: Site configuration for educational content
- **Fields**:
  - sidebarConfig (object): Navigation structure for the module
  - siteConfig (object): Docusaurus site configuration (docusaurus.config.js)
  - themeConfig (object): Styling and theme settings
  - frontmatterTemplate (object): Standard frontmatter for all chapters
- **Relationships**: Governs the presentation of all Chapter Content
- **Validation rules**: Must follow Docusaurus v3.x standards