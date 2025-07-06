# Gemini CLI Prompt: Advanced PDF Test Analysis Tool

## Project Overview
Create a comprehensive PDF-based test analysis and delivery system for JEE Main/Advanced examinations. The tool should parse PDF test papers, extract questions with images, classify question types, provide appropriate interfaces, and evaluate answers with accurate marking schemes.

## Core Requirements

## Core Requirements (Maximum Accuracy Configuration)

### 1. Resource-Efficient PDF Processing Engine
```
Build a lightweight but accurate PDF analysis module:
- Primary: PyMuPDF for fast text extraction
- Secondary: pdfplumber for complex layouts
- OCR: Tesseract with mathematical symbol recognition
- Image processing: OpenCV for layout analysis (lightweight)
- Memory-efficient: Process PDFs page by page
- Caching: Store processed results to avoid reprocessing
- Fallback methods: Rule-based parsing when ML fails
- Local processing: No external API dependencies
- Optimized: Works within 512MB RAM constraints
```

### 2. Lightweight Content Classification (Hybrid Approach)
```
Implement efficient classification using minimal resources:
- Primary: Rule-based classification using regex and patterns
- Secondary: Small pre-trained models (DistilBERT, 66MB) for edge cases
- Pattern matching: Extensive regex patterns for question types
- Layout analysis: Simple computer vision using OpenCV
- Confidence scoring: Based on pattern matching certainty
- Fallback system: Multiple detection methods in priority order
- Memory efficient: Load models on-demand, unload after use
- Local inference: No external API calls required
- Training: Use few-shot learning on small datasets
- Validation: Cross-check with answer key formats
```

### 3. Ultra-Precise Question Type Detection
```
Implement multi-layered detection with validation:

SCQ (Single Correct Questions):
- Pattern matching with regex and NLP models
- Validate exactly 4 options exist
- Check for standard formatting patterns
- Confirm no "multiple correct" indicators
- Cross-validate with answer key format

MCQ (Multiple Correct Questions):
- Detect instruction keywords: "One or more", "Multiple correct"
- Validate 4 options with multiple possible correct answers
- Check for specific instruction patterns
- Cross-validate with answer key showing multiple correct options

Integer Type Questions:
- Detect absence of options (A), (B), (C), (D)
- Identify numerical answer requirements
- Check for range specifications (0-9999)
- Validate question asks for numerical value
- Confirm answer key has integer format

Match the Column:
- Detect two-column structure with headers
- Identify matching relationships
- Validate option combinations format
- Check for proper column labeling
- Cross-validate with complex answer key format

Confidence Scoring:
- Each detection method provides confidence score
- Require >95% confidence for automatic classification
- Flag uncertain cases for manual review
- Implement validation across multiple detection methods
```

### 4. Marking Scheme Implementation
```
Implement detailed marking schemes:

JEE Main:
- SCQ: +4 correct, -1 incorrect, 0 unattempted
- MCQ: +4 all correct, +1 to +3 partial, -2 any incorrect, 0 unattempted
- Integer: +4 correct, -1 incorrect, 0 unattempted

JEE Advanced:
- SCQ: +3 correct, -1 incorrect, 0 unattempted
- MCQ: +4 all correct, +1 to +3 partial, -2 any incorrect, 0 unattempted
- Match Column: +3 perfect match, +1 per correct match, -1 any incorrect, 0 unattempted
- Integer: +3 correct, 0 incorrect, 0 unattempted

Implement proportional scoring for MCQ partial answers
Handle section-wise different marking schemes
```

### 5. User Interface Components
```
Design responsive interfaces:
- Question display with proper formatting and image rendering
- Radio buttons for SCQ
- Checkboxes for MCQ
- Number input box for Integer type (0-9999 range)
- Interactive matching interface for Match the Column (drag-drop or dropdown)
- MathJax/KaTeX integration for mathematical expressions
- Image zoom and pan functionality
- Question navigation panel
- Timer display (overall and per question)
- Save and review functionality
```

### 6. Efficient PDF Analysis Features (Resource-Optimized)
```
Implement smart analysis within resource constraints:
- OCR: Tesseract with custom configurations for mathematical content
- Layout analysis: OpenCV-based structure detection
- Mathematical equations: Pattern-based recognition + symbol libraries
- Image processing: Efficient extraction and compression
- Memory management: Process in chunks, clear cache regularly
- Progressive processing: Start with simple methods, escalate if needed
- Smart caching: Store intermediate results to avoid reprocessing
- Batch processing: Handle multiple elements efficiently
- Quality assessment: Simple confidence scoring
- Error handling: Graceful degradation with fallback methods
```

### 7. Answer Key Processing
```
Build flexible answer key parser:
- Support multiple formats (A,B,C,D / 1,2,3,4 / numerical)
- Handle answer keys in separate PDFs or text files
- Parse complex answer formats for Match the Column
- Validate answer key completeness
- Support partial answer keys
- Handle answer key corrections/updates
```

### 8. Evaluation Engine
```
Create comprehensive evaluation system:
- Compare user answers with answer keys
- Apply appropriate marking schemes based on question type
- Calculate section-wise and overall scores
- Generate detailed performance analytics
- Track time spent per question
- Provide question-wise analysis
- Generate percentile and rank estimation
- Export results in multiple formats
```

### 9. Learning and Adaptation
```
Implement adaptive learning:
- Learn from user feedback on question detection accuracy
- Improve classification models with new PDF samples
- Adapt to different PDF formats and layouts
- Store successful parsing patterns
- Continuously improve instruction filtering
- Learn from answer key parsing errors
```

### 10. Technical Architecture (Resource-Constrained Optimization)
```
Structure for maximum accuracy within platform constraints:

**For Windows 11 Local Deployment:**
- Backend: Python with FastAPI/Flask (lightweight)
- PDF Processing: PyMuPDF + pdfplumber (no external API calls)
- ML Models: Lightweight models (DistilBERT, TinyBERT) with local inference
- Database: SQLite for simplicity, no external database required
- OCR: Tesseract (local) + easyOCR for mathematical content
- Storage: Local file system with organized folder structure
- Memory: Optimized for 8-16GB RAM systems

**For Replit Free Tier (512MB RAM, 0.5 CPU):**
- Minimal dependencies, on-demand model loading
- Process PDFs in chunks to manage memory
- Use rule-based detection primarily, ML as fallback
- SQLite database, no external services
- Compress/cache processed results
- Implement smart caching to avoid reprocessing

**For Render Free Tier (512MB RAM, shared CPU):**
- Stateless design with minimal memory footprint
- Use serverless approach with cold starts
- Implement request queuing for resource management
- SQLite with persistent storage
- Optimize for 15-minute timeout limits
- Use background processing for large PDFs

**Lightweight ML Approach:**
- Pre-trained lightweight models (50-100MB total)
- Rule-based classification as primary method
- Pattern matching for question type detection
- Simple confidence scoring without heavy computation
- Local model inference, no API calls
```

## Implementation Steps

### Phase 1: Foundation
```
1. Set up basic PDF parsing with text and image extraction
2. Implement simple question detection algorithms
3. Create basic question type classification
4. Build minimal UI for question display
```

### Phase 2: Intelligence
```
1. Implement ML-based content classification
2. Add advanced question type detection
3. Build instruction filtering system
4. Create answer key parsing module
```

### Phase 3: User Experience
```
1. Develop complete UI with all question types
2. Implement timer and navigation features
3. Add mathematical expression rendering
4. Create responsive design for all devices
```

### Phase 4: Evaluation
```
1. Implement all marking schemes
2. Build comprehensive evaluation engine
3. Add performance analytics
4. Create result export functionality
```

### Phase 5: Advanced Features
```
1. Add machine learning improvements
2. Implement adaptive learning
3. Add multi-language support
4. Create admin panel for test management
```

## Success Metrics (High Accuracy Focus)
```
- Accuracy of question extraction: >99.5%
- Question type classification: >99%
- Instruction filtering accuracy: >99.8%
- Answer key parsing accuracy: >99.9%
- Mathematical expression rendering: 100% accuracy
- Marking scheme calculation: 100% accuracy
- Image extraction and preservation: >99.5%
- Layout structure maintenance: >99%
```

## Testing Strategy (Maximum Accuracy Validation)
```
1. Extensive testing with JEE papers from last 15 years (500+ papers)
2. Manual verification of every extraction on sample set
3. Cross-validation with official answer keys and solutions
4. Edge case testing with corrupted/low-quality PDFs
5. Mathematical expression accuracy testing with expert review
6. A/B testing between different parsing methods
7. Regression testing with previously processed papers
8. Human expert validation for classification accuracy
9. Performance benchmarking against manual extraction
10. Error analysis and iterative improvement cycles
```

## Deliverables
```
1. Complete source code with documentation
2. Trained ML models for content classification
3. Database schema and setup scripts
4. API documentation
5. User manual and admin guide
6. Test suite with sample PDFs
7. Deployment guide and Docker containers
```

## Platform-Specific Deployment Guides

### Windows 11 Local Deployment
```
System Requirements:
- Windows 11 with 8GB+ RAM
- Python 3.8+ with pip
- 2GB free disk space

Installation Steps:
1. Install Python dependencies: PyMuPDF, pdfplumber, tesseract, OpenCV
2. Download and install Tesseract OCR
3. Set up SQLite database
4. Configure local file storage
5. Run with: python app.py

Optimizations:
- Use virtual environment for dependency isolation
- Enable Windows Defender exclusions for faster file access
- Configure page file for memory management
- Use SSD for faster PDF processing
```

### Replit Free Tier Deployment
```
Resource Limits:
- 512MB RAM, 0.5 CPU core
- 1GB storage, 100MB max file size
- 10-hour monthly limit

Implementation Strategy:
- Use replit.nix for dependency management
- Implement request queuing to manage resources
- Process PDFs in small chunks
- Use background tasks for heavy processing
- Cache results aggressively
- Implement auto-sleep to conserve resources

Code Structure:
- main.py: Lightweight FastAPI app
- processor.py: PDF processing module
- models.py: Lightweight ML models
- utils.py: Helper functions
```

### Render Free Tier Deployment
```
Resource Limits:
- 512MB RAM, shared CPU
- 15-minute request timeout
- Cold start delays

Implementation Strategy:
- Use Dockerfile for consistent deployment
- Implement health checks
- Use persistent storage for processed results
- Optimize for cold starts
- Queue long-running processes
- Use webhooks for async processing

Deployment Files:
- Dockerfile: Container configuration
- render.yaml: Service configuration
- requirements.txt: Python dependencies
- startup.sh: Initialization script
```

## Resource Optimization Strategies
```
Memory Management:
- Process PDFs page by page
- Unload models after use
- Use generators instead of lists
- Implement smart caching
- Clear memory regularly

Performance Optimization:
- Lazy loading of components
- Async processing where possible
- Efficient data structures
- Minimize I/O operations
- Use compression for storage

Model Optimization:
- Use quantized models
- Implement model pruning
- Use ONNX for faster inference
- Load models on-demand
- Cache model predictions
```

This tool should be production-ready, scalable, and capable of handling thousands of concurrent users while maintaining accuracy and performance.