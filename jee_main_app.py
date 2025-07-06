#!/usr/bin/env python3
"""
JEE Test Analysis System - Main Application
Advanced PDF Test Analysis Tool for JEE Main/Advanced examinations
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import asyncio
from datetime import datetime, timedelta

# Core dependencies
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# PDF Processing
import fitz  # PyMuPDF
import pdfplumber
import cv2
import numpy as np
from PIL import Image
import pytesseract

# ML and NLP
import re
from collections import defaultdict
import pickle
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project structure
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create directories
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR, TEMPLATES_DIR, STATIC_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data models
@dataclass
class Question:
    """Represents a parsed question"""
    id: str
    type: str  # SCQ, MCQ, INTEGER, MATCH_COLUMN
    text: str
    options: List[str]
    images: List[str]
    section: str
    confidence: float
    page_number: int
    position: Dict[str, int]
    metadata: Dict

@dataclass
class TestPaper:
    """Represents a complete test paper"""
    id: str
    title: str
    questions: List[Question]
    sections: List[str]
    total_questions: int
    metadata: Dict
    created_at: datetime

@dataclass
class Answer:
    """Represents a user's answer"""
    question_id: str
    answer: str
    time_spent: int
    timestamp: datetime

@dataclass
class TestResult:
    """Represents test evaluation results"""
    test_id: str
    user_id: str
    total_score: float
    section_scores: Dict[str, float]
    question_results: Dict[str, Dict]
    time_taken: int
    accuracy: float
    percentile: float

# Database Manager
class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str = "jee_test_system.db"):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS test_papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sections TEXT,
                    total_questions INTEGER,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    test_id TEXT,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    options TEXT,
                    images TEXT,
                    section TEXT,
                    confidence REAL,
                    page_number INTEGER,
                    position TEXT,
                    metadata TEXT,
                    FOREIGN KEY (test_id) REFERENCES test_papers (id)
                );
                
                CREATE TABLE IF NOT EXISTS answer_keys (
                    id TEXT PRIMARY KEY,
                    test_id TEXT,
                    question_id TEXT,
                    correct_answer TEXT NOT NULL,
                    explanation TEXT,
                    FOREIGN KEY (test_id) REFERENCES test_papers (id),
                    FOREIGN KEY (question_id) REFERENCES questions (id)
                );
                
                CREATE TABLE IF NOT EXISTS user_answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT,
                    question_id TEXT,
                    user_id TEXT,
                    answer TEXT,
                    time_spent INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES test_papers (id),
                    FOREIGN KEY (question_id) REFERENCES questions (id)
                );
                
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT,
                    user_id TEXT,
                    total_score REAL,
                    section_scores TEXT,
                    question_results TEXT,
                    time_taken INTEGER,
                    accuracy REAL,
                    percentile REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES test_papers (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_test_questions ON questions(test_id);
                CREATE INDEX IF NOT EXISTS idx_user_answers ON user_answers(test_id, user_id);
                CREATE INDEX IF NOT EXISTS idx_test_results ON test_results(test_id, user_id);
            """)
            conn.commit()

# PDF Processing Engine
class PDFProcessor:
    """Handles PDF parsing and content extraction"""
    
    def __init__(self):
        self.cache = {}
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for question type detection"""
        return {
            'scq_indicators': [
                r'(?i)single\s+correct',
                r'(?i)only\s+one\s+correct',
                r'(?i)exactly\s+one\s+correct',
                r'\(A\)\s*.*\(B\)\s*.*\(C\)\s*.*\(D\)\s*.*$'
            ],
            'mcq_indicators': [
                r'(?i)one\s+or\s+more\s+correct',
                r'(?i)multiple\s+correct',
                r'(?i)more\s+than\s+one\s+correct',
                r'(?i)which\s+of\s+the\s+following\s+are\s+correct'
            ],
            'integer_indicators': [
                r'(?i)answer\s+is\s+an?\s+integer',
                r'(?i)numerical\s+value',
                r'(?i)range\s+0\s*-\s*9999',
                r'(?i)answer\s+to\s+the\s+nearest\s+integer'
            ],
            'match_column_indicators': [
                r'(?i)match\s+the\s+column',
                r'(?i)match\s+list\s+I\s+with\s+list\s+II',
                r'Column\s+I.*Column\s+II',
                r'List\s+I.*List\s+II'
            ]
        }
    
    def extract_text_and_images(self, pdf_path: str) -> Tuple[List[str], List[str]]:
        """Extract text and images from PDF"""
        text_pages = []
        image_paths = []
        
        # Use PyMuPDF for text and images
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                text_pages.append(text)
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_path = PROCESSED_DIR / f"img_{page_num}_{img_index}.png"
                        pix.save(str(img_path))
                        image_paths.append(str(img_path))
                    pix = None
        
        return text_pages, image_paths
    
    def detect_question_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Detect question boundaries in text"""
        boundaries = []
        
        # Pattern for question numbers
        question_pattern = r'(?:^|\n)\s*(\d+)\s*[.)]'
        matches = list(re.finditer(question_pattern, text, re.MULTILINE))
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            boundaries.append((start, end))
        
        return boundaries
    
    def classify_question_type(self, question_text: str) -> Tuple[str, float]:
        """Classify question type with confidence score"""
        text_lower = question_text.lower()
        
        # Check for each type with confidence scoring
        type_scores = {
            'SCQ': 0.0,
            'MCQ': 0.0,
            'INTEGER': 0.0,
            'MATCH_COLUMN': 0.0
        }
        
        # SCQ detection
        for pattern in self.patterns['scq_indicators']:
            if re.search(pattern, question_text, re.IGNORECASE):
                type_scores['SCQ'] += 0.3
        
        # Count options
        option_pattern = r'\([A-D]\)'
        options_count = len(re.findall(option_pattern, question_text))
        if options_count == 4:
            type_scores['SCQ'] += 0.4
        
        # MCQ detection
        for pattern in self.patterns['mcq_indicators']:
            if re.search(pattern, question_text, re.IGNORECASE):
                type_scores['MCQ'] += 0.6
        
        # Integer detection
        for pattern in self.patterns['integer_indicators']:
            if re.search(pattern, question_text, re.IGNORECASE):
                type_scores['INTEGER'] += 0.5
        
        if options_count == 0:
            type_scores['INTEGER'] += 0.3
        
        # Match column detection
        for pattern in self.patterns['match_column_indicators']:
            if re.search(pattern, question_text, re.IGNORECASE):
                type_scores['MATCH_COLUMN'] += 0.7
        
        # Determine best type
        best_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[best_type]
        
        # Default to SCQ if confidence is too low
        if confidence < 0.3:
            best_type = 'SCQ'
            confidence = 0.3
        
        return best_type, min(confidence, 1.0)
    
    def extract_options(self, question_text: str, question_type: str) -> List[str]:
        """Extract options from question text"""
        options = []
        
        if question_type in ['SCQ', 'MCQ']:
            # Extract options (A), (B), (C), (D)
            option_pattern = r'\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|\n|$)'
            matches = re.findall(option_pattern, question_text, re.DOTALL)
            
            for letter, text in matches:
                options.append(f"({letter}) {text.strip()}")
        
        elif question_type == 'MATCH_COLUMN':
            # Extract column items
            column_pattern = r'(?:Column\s+I|List\s+I)(.*?)(?:Column\s+II|List\s+II)(.*?)(?:\n\n|$)'
            match = re.search(column_pattern, question_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                col1, col2 = match.groups()
                options.append(f"Column I: {col1.strip()}")
                options.append(f"Column II: {col2.strip()}")
        
        return options
    
    def process_pdf(self, pdf_path: str, test_id: str) -> TestPaper:
        """Process complete PDF and extract questions"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text and images
        text_pages, image_paths = self.extract_text_and_images(pdf_path)
        
        # Combine all text
        full_text = "\n".join(text_pages)
        
        # Detect questions
        boundaries = self.detect_question_boundaries(full_text)
        questions = []
        
        for i, (start, end) in enumerate(boundaries):
            question_text = full_text[start:end].strip()
            
            if len(question_text) < 20:  # Skip very short text
                continue
            
            # Classify question type
            q_type, confidence = self.classify_question_type(question_text)
            
            # Extract options
            options = self.extract_options(question_text, q_type)
            
            # Create question object
            question = Question(
                id=f"{test_id}_q{i+1}",
                type=q_type,
                text=question_text,
                options=options,
                images=[],  # Will be populated later
                section="General",  # Will be detected later
                confidence=confidence,
                page_number=0,  # Will be calculated
                position={"start": start, "end": end},
                metadata={}
            )
            
            questions.append(question)
        
        # Create test paper
        test_paper = TestPaper(
            id=test_id,
            title=f"Test Paper {test_id}",
            questions=questions,
            sections=["Physics", "Chemistry", "Mathematics"],
            total_questions=len(questions),
            metadata={"pdf_path": pdf_path, "images": image_paths},
            created_at=datetime.now()
        )
        
        logger.info(f"Processed {len(questions)} questions from PDF")
        return test_paper

# Question Type Classifier
class QuestionClassifier:
    """Advanced question type classification"""
    
    def __init__(self):
        self.confidence_threshold = 0.95
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict:
        """Load validation rules for each question type"""
        return {
            'SCQ': {
                'required_options': 4,
                'indicators': ['single correct', 'only one correct'],
                'answer_format': r'^[A-D]$'
            },
            'MCQ': {
                'required_options': 4,
                'indicators': ['one or more correct', 'multiple correct'],
                'answer_format': r'^[A-D,\s]+$'
            },
            'INTEGER': {
                'required_options': 0,
                'indicators': ['integer', 'numerical value'],
                'answer_format': r'^\d+$'
            },
            'MATCH_COLUMN': {
                'required_options': 2,
                'indicators': ['match', 'column'],
                'answer_format': r'^[A-D]-[1-4]'
            }
        }
    
    def validate_classification(self, question: Question, answer_key: str = None) -> bool:
        """Validate question classification"""
        rules = self.validation_rules.get(question.type, {})
        
        # Check options count
        required_options = rules.get('required_options', 0)
        if question.type in ['SCQ', 'MCQ'] and len(question.options) != required_options:
            return False
        
        # Check answer key format if provided
        if answer_key and 'answer_format' in rules:
            if not re.match(rules['answer_format'], answer_key):
                return False
        
        return True

# Marking Scheme Engine
class MarkingScheme:
    """Handles marking schemes for different question types"""
    
    JEE_MAIN_SCHEME = {
        'SCQ': {'correct': 4, 'incorrect': -1, 'unattempted': 0},
        'MCQ': {'all_correct': 4, 'partial': [1, 2, 3], 'incorrect': -2, 'unattempted': 0},
        'INTEGER': {'correct': 4, 'incorrect': -1, 'unattempted': 0}
    }
    
    JEE_ADVANCED_SCHEME = {
        'SCQ': {'correct': 3, 'incorrect': -1, 'unattempted': 0},
        'MCQ': {'all_correct': 4, 'partial': [1, 2, 3], 'incorrect': -2, 'unattempted': 0},
        'MATCH_COLUMN': {'perfect': 3, 'partial': 1, 'incorrect': -1, 'unattempted': 0},
        'INTEGER': {'correct': 3, 'incorrect': 0, 'unattempted': 0}
    }
    
    def __init__(self, scheme_type: str = "JEE_MAIN"):
        self.scheme = self.JEE_MAIN_SCHEME if scheme_type == "JEE_MAIN" else self.JEE_ADVANCED_SCHEME
    
    def calculate_score(self, question_type: str, correct_answer: str, user_answer: str) -> float:
        """Calculate score for a question"""
        if not user_answer:  # Unattempted
            return self.scheme[question_type]['unattempted']
        
        if question_type == 'SCQ':
            return self.scheme['SCQ']['correct'] if user_answer == correct_answer else self.scheme['SCQ']['incorrect']
        
        elif question_type == 'MCQ':
            correct_options = set(correct_answer.split(','))
            user_options = set(user_answer.split(','))
            
            if user_options == correct_options:
                return self.scheme['MCQ']['all_correct']
            elif user_options.issubset(correct_options):
                # Partial credit
                return self.scheme['MCQ']['partial'][len(user_options) - 1]
            else:
                return self.scheme['MCQ']['incorrect']
        
        elif question_type == 'INTEGER':
            return self.scheme['INTEGER']['correct'] if user_answer == correct_answer else self.scheme['INTEGER']['incorrect']
        
        elif question_type == 'MATCH_COLUMN':
            # Complex matching logic
            correct_matches = correct_answer.split(',')
            user_matches = user_answer.split(',')
            
            if set(user_matches) == set(correct_matches):
                return self.scheme['MATCH_COLUMN']['perfect']
            else:
                # Count correct matches
                correct_count = len(set(user_matches) & set(correct_matches))
                return correct_count * self.scheme['MATCH_COLUMN']['partial']
        
        return 0

# Initialize FastAPI app
app = FastAPI(title="JEE Test Analysis System", version="1.0.0")

# Initialize components
db_manager = DatabaseManager()
pdf_processor = PDFProcessor()
question_classifier = QuestionClassifier()
marking_scheme = MarkingScheme()

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# API Models
class TestUpload(BaseModel):
    title: str
    scheme_type: str = "JEE_MAIN"

class UserAnswer(BaseModel):
    question_id: str
    answer: str
    time_spent: int

class TestSubmission(BaseModel):
    test_id: str
    user_id: str
    answers: List[UserAnswer]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-test")
async def upload_test(
    file: UploadFile = File(...),
    title: str = Form(...),
    scheme_type: str = Form("JEE_MAIN")
):
    """Upload and process test PDF"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Generate test ID
        test_id = f"test_{int(datetime.now().timestamp())}"
        
        # Process PDF
        test_paper = pdf_processor.process_pdf(str(file_path), test_id)
        test_paper.title = title
        
        # Save to database
        with db_manager.get_connection() as conn:
            conn.execute(
                """INSERT INTO test_papers (id, title, file_path, sections, total_questions, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (test_id, title, str(file_path), json.dumps(test_paper.sections),
                 test_paper.total_questions, json.dumps(test_paper.metadata))
            )
            
            # Save questions
            for question in test_paper.questions:
                conn.execute(
                    """INSERT INTO questions (id, test_id, type, text, options, images, section, confidence, page_number, position, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (question.id, test_id, question.type, question.text,
                     json.dumps(question.options), json.dumps(question.images),
                     question.section, question.confidence, question.page_number,
                     json.dumps(question.position), json.dumps(question.metadata))
                )
            
            conn.commit()
        
        return {"test_id": test_id, "message": "Test uploaded successfully", "questions": len(test_paper.questions)}
    
    except Exception as e:
        logger.error(f"Error uploading test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/{test_id}")
async def get_test(test_id: str):
    """Get test details"""
    try:
        with db_manager.get_connection() as conn:
            # Get test paper
            test_row = conn.execute(
                "SELECT * FROM test_papers WHERE id = ?", (test_id,)
            ).fetchone()
            
            if not test_row:
                raise HTTPException(status_code=404, detail="Test not found")
            
            # Get questions
            questions = conn.execute(
                "SELECT * FROM questions WHERE test_id = ? ORDER BY id",
                (test_id,)
            ).fetchall()
            
            return {
                "test": dict(test_row),
                "questions": [dict(q) for q in questions]
            }
    
    except Exception as e:
        logger.error(f"Error getting test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-test")
async def submit_test(submission: TestSubmission):
    """Submit test answers and calculate results"""
    try:
        # Get test and questions
        with db_manager.get_connection() as conn:
            questions = conn.execute(
                "SELECT * FROM questions WHERE test_id = ?",
                (submission.test_id,)
            ).fetchall()
            
            # Get answer keys (if available)
            answer_keys = {}
            for row in conn.execute(
                "SELECT question_id, correct_answer FROM answer_keys WHERE test_id = ?",
                (submission.test_id,)
            ).fetchall():
                answer_keys[row['question_id']] = row['correct_answer']
            
            # Save user answers
            for answer in submission.answers:
                conn.execute(
                    """INSERT INTO user_answers (test_id, question_id, user_id, answer, time_spent)
                       VALUES (?, ?, ?, ?, ?)""",
                    (submission.test_id, answer.question_id, submission.user_id,
                     answer.answer, answer.time_spent)
                )
            
            # Calculate results if answer keys are available
            if answer_keys:
                total_score = 0
                question_results = {}
                
                for question in questions:
                    q_id = question['id']
                    q_type = question['type']
                    
                    # Find user answer
                    user_answer = ""
                    for ans in submission.answers:
                        if ans.question_id == q_id:
                            user_answer = ans.answer
                            break
                    
                    # Calculate score
                    if q_id in answer_keys:
                        score = marking_scheme.calculate_score(
                            q_type, answer_keys[q_id], user_answer
                        )
                        total_score += score
                        
                        question_results[q_id] = {
                            'score': score,
                            'correct_answer': answer_keys[q_id],
                            'user_answer': user_answer
                        }
                
                # Save results
                conn.execute(
                    """INSERT INTO test_results (test_id, user_id, total_score, question_results, time_taken)
                       VALUES (?, ?, ?, ?, ?)""",
                    (submission.test_id, submission.user_id, total_score,
                     json.dumps(question_results), sum(a.time_spent for a in submission.answers))
                )
            
            conn.commit()
        
        return {"message": "Test submitted successfully", "total_score": total_score if answer_keys else "Pending"}
    
    except Exception as e:
        logger.error(f"Error submitting test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{test_id}/{user_id}")
async def get_results(test_id: str, user_id: str):
    """Get test results"""
    try:
        with db_manager.get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM test_results WHERE test_id = ? AND user_id = ?",
                (test_id, user_id)
            ).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Results not found")
            
            return dict(result)
    
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)