#!/usr/bin/env python3
"""
JEE Test Analysis System - Complete Implementation
Advanced PDF Test Analysis Tool for JEE Main/Advanced examinations
Enhanced with ML-based question classification and comprehensive marking schemes
"""

import os
import sys
import logging
import hashlib
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict
import sqlite3
import pickle
import uuid

# Core dependencies
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# PDF Processing
try:
    import fitz  # PyMuPDF
    import pdfplumber
    HAS_PDF_LIBS = True
except ImportError:
    HAS_PDF_LIBS = False
    logging.warning("PDF processing libraries not available")

# Image Processing
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    HAS_CV_LIBS = True
except ImportError:
    HAS_CV_LIBS = False
    logging.warning("Computer vision libraries not available")

# ML Dependencies
try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    logging.warning("ML libraries not available - using fallback classification")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jee_system.log'),
        logging.StreamHandler()
    ]
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
IMAGES_DIR = STATIC_DIR / "images"

# Create directories
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR, TEMPLATES_DIR, STATIC_DIR, IMAGES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Configuration
class Config:
    # System limits
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_QUESTIONS_PER_TEST = 200
    CONFIDENCE_THRESHOLD = 0.85
    
    # ML settings
    USE_ML_CLASSIFICATION = HAS_ML_LIBS
    MODEL_CACHE_SIZE = 100
    
    # Processing settings
    ENABLE_OCR = HAS_CV_LIBS
    IMAGE_DPI = 300
    OCR_CONFIDENCE_THRESHOLD = 60
    
    # Marking schemes
    DEFAULT_SCHEME = "JEE_MAIN"
    ENABLE_PARTIAL_MARKING = True
    
    # UI settings
    QUESTIONS_PER_PAGE = 1
    AUTO_SAVE_INTERVAL = 30  # seconds
    TEST_TIME_LIMIT = 180  # minutes

config = Config()

# Enhanced Data Models
@dataclass
class QuestionImage:
    """Image associated with a question"""
    path: str
    type: str  # 'diagram', 'graph', 'table', 'equation'
    confidence: float
    position: Dict[str, int]
    alt_text: str = ""

@dataclass
class Question:
    """Enhanced question model with ML features"""
    id: str
    test_id: str
    type: str  # SCQ, MCQ, INTEGER, MATCH_COLUMN
    text: str
    options: List[str]
    images: List[QuestionImage]
    section: str
    subject: str  # Physics, Chemistry, Mathematics
    topic: str
    difficulty: str  # Easy, Medium, Hard
    confidence: float
    page_number: int
    position: Dict[str, int]
    metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'test_id': self.test_id,
            'type': self.type,
            'text': self.text,
            'options': self.options,
            'images': [asdict(img) for img in self.images],
            'section': self.section,
            'subject': self.subject,
            'topic': self.topic,
            'difficulty': self.difficulty,
            'confidence': self.confidence,
            'page_number': self.page_number,
            'position': self.position,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

@dataclass
class TestPaper:
    """Enhanced test paper model"""
    id: str
    title: str
    description: str
    questions: List[Question]
    sections: List[str]
    subjects: List[str]
    total_questions: int
    time_limit: int  # minutes
    marking_scheme: str
    metadata: Dict[str, Any]
    created_at: datetime
    
    def get_section_questions(self, section: str) -> List[Question]:
        """Get questions for a specific section"""
        return [q for q in self.questions if q.section == section]
    
    def get_subject_questions(self, subject: str) -> List[Question]:
        """Get questions for a specific subject"""
        return [q for q in self.questions if q.subject == subject]

@dataclass
class Answer:
    """User answer with enhanced tracking"""
    question_id: str
    answer: str
    time_spent: int
    attempts: int
    confidence: float
    marked_for_review: bool
    timestamp: datetime

@dataclass
class TestResult:
    """Comprehensive test results"""
    test_id: str
    user_id: str
    total_score: float
    max_score: float
    percentage: float
    section_scores: Dict[str, Dict[str, float]]
    subject_scores: Dict[str, Dict[str, float]]
    question_results: Dict[str, Dict[str, Any]]
    time_taken: int
    accuracy: float
    attempted_questions: int
    correct_answers: int
    partial_answers: int
    incorrect_answers: int
    rank: int
    percentile: float
    analysis: Dict[str, Any]
    created_at: datetime

# Advanced PDF Processing Engine
class EnhancedPDFProcessor:
    """Advanced PDF processing with ML-enhanced question detection"""
    
    def __init__(self):
        self.cache = {}
        self.ocr_cache = {}
        self.patterns = self._load_enhanced_patterns()
        self.ml_classifier = None
        
        if config.USE_ML_CLASSIFICATION:
            self.ml_classifier = MLQuestionClassifier()
    
    def _load_enhanced_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive regex patterns for question detection"""
        return {
            'question_boundaries': {
                'numbered': [
                    r'(?:^|\n)\s*(\d+)\s*[.)]',
                    r'(?:^|\n)\s*Q\s*[.-]?\s*(\d+)',
                    r'(?:^|\n)\s*Question\s+(\d+)',
                    r'(?:^|\n)\s*(\d+)\s*[:-]'
                ],
                'sectioned': [
                    r'(?:^|\n)\s*(\d+)\s*[.)].*?(?=\n\s*\d+\s*[.)]|\Z)',
                    r'(?:Section\s+[A-Z].*?)?(?:^|\n)\s*(\d+)\s*[.)]'
                ]
            },
            'question_types': {
                'scq': {
                    'indicators': [
                        r'(?i)single\s+correct\s+(?:answer|option)',
                        r'(?i)only\s+one\s+correct\s+(?:answer|option)',
                        r'(?i)exactly\s+one\s+correct',
                        r'(?i)choose\s+the\s+correct\s+(?:answer|option)',
                        r'(?i)select\s+the\s+correct\s+(?:answer|option)'
                    ],
                    'options_pattern': r'\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|\n\s*\d+[.)]|\Z)',
                    'required_options': 4
                },
                'mcq': {
                    'indicators': [
                        r'(?i)one\s+or\s+more\s+correct',
                        r'(?i)multiple\s+correct\s+(?:answers|options)',
                        r'(?i)more\s+than\s+one\s+correct',
                        r'(?i)which\s+of\s+the\s+following\s+are\s+correct',
                        r'(?i)select\s+all\s+correct',
                        r'(?i)mark\s+all\s+correct'
                    ],
                    'options_pattern': r'\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|\n\s*\d+[.)]|\Z)',
                    'required_options': 4
                },
                'integer': {
                    'indicators': [
                        r'(?i)answer\s+is\s+an?\s+integer',
                        r'(?i)numerical\s+value',
                        r'(?i)range\s+0\s*[-to]\s*9999',
                        r'(?i)answer\s+to\s+the\s+nearest\s+integer',
                        r'(?i)find\s+the\s+value\s+of',
                        r'(?i)calculate\s+the\s+value'
                    ],
                    'no_options': True,
                    'answer_range': (0, 9999)
                },
                'match_column': {
                    'indicators': [
                        r'(?i)match\s+the\s+column',
                        r'(?i)match\s+list\s+[I1]\s+with\s+list\s+[I2]',
                        r'(?i)column\s+[I1].*column\s+[I2]',
                        r'(?i)list\s+[I1].*list\s+[I2]',
                        r'(?i)match\s+the\s+following'
                    ],
                    'column_pattern': r'(?:Column\s+[I1]|List\s+[I1])(.*?)(?:Column\s+[I2]|List\s+[I2])(.*?)(?:\n\n|$)',
                    'item_pattern': r'\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|\n|$)'
                }
            },
            'subjects': {
                'physics': [
                    r'(?i)mechanics', r'(?i)thermodynamics', r'(?i)electromagnetism',
                    r'(?i)optics', r'(?i)modern\s+physics', r'(?i)waves',
                    r'(?i)oscillations', r'(?i)kinematics', r'(?i)dynamics'
                ],
                'chemistry': [
                    r'(?i)organic', r'(?i)inorganic', r'(?i)physical\s+chemistry',
                    r'(?i)coordination', r'(?i)electrochemistry', r'(?i)chemical\s+bonding',
                    r'(?i)periodic\s+table', r'(?i)thermochemistry'
                ],
                'mathematics': [
                    r'(?i)calculus', r'(?i)algebra', r'(?i)geometry',
                    r'(?i)trigonometry', r'(?i)statistics', r'(?i)probability',
                    r'(?i)coordinate\s+geometry', r'(?i)differential\s+equations'
                ]
            }
        }
    
    def process_pdf_advanced(self, pdf_path: str, test_id: str) -> TestPaper:
        """Advanced PDF processing with ML enhancement"""
        logger.info(f"Processing PDF with advanced features: {pdf_path}")
        
        try:
            # Extract text and images with multiple methods
            text_data = self._extract_text_multi_method(pdf_path)
            images = self._extract_and_process_images(pdf_path, test_id)
            
            # Detect and extract questions
            questions = self._detect_questions_advanced(text_data, images, test_id)
            
            # Classify questions with ML
            if self.ml_classifier:
                questions = self._enhance_with_ml_classification(questions)
            
            # Create test paper
            test_paper = TestPaper(
                id=test_id,
                title=f"Test Paper {test_id}",
                description="Auto-generated from PDF",
                questions=questions,
                sections=self._detect_sections(questions),
                subjects=self._detect_subjects(questions),
                total_questions=len(questions),
                time_limit=config.TEST_TIME_LIMIT,
                marking_scheme=config.DEFAULT_SCHEME,
                metadata={
                    'pdf_path': pdf_path,
                    'processing_time': time.time(),
                    'confidence_stats': self._calculate_confidence_stats(questions)
                },
                created_at=datetime.now()
            )
            
            logger.info(f"Successfully processed {len(questions)} questions")
            return test_paper
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _extract_text_multi_method(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using multiple methods for better accuracy"""
        text_data = []
        
        if not HAS_PDF_LIBS:
            raise Exception("PDF processing libraries not available")
        
        try:
            # Method 1: PyMuPDF
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    blocks = page.get_text("blocks")
                    
                    text_data.append({
                        'page': page_num,
                        'text': text,
                        'blocks': blocks,
                        'method': 'pymupdf'
                    })
            
            # Method 2: pdfplumber for better table detection
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    if page_num < len(text_data):
                        text_data[page_num]['pdfplumber_text'] = page.extract_text()
                        text_data[page_num]['tables'] = page.extract_tables()
            
            return text_data
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def _extract_and_process_images(self, pdf_path: str, test_id: str) -> List[QuestionImage]:
        """Extract and process images with OCR"""
        images = []
        
        if not HAS_PDF_LIBS or not HAS_CV_LIBS:
            return images
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    image_list = page.get_images(full=True)
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_filename = f"{test_id}_p{page_num}_i{img_index}.png"
                                img_path = IMAGES_DIR / img_filename
                                
                                pix.save(str(img_path))
                                
                                # Process image with OCR
                                processed_img = self._process_image_ocr(str(img_path))
                                
                                question_image = QuestionImage(
                                    path=str(img_path),
                                    type=self._classify_image_type(str(img_path)),
                                    confidence=processed_img.get('confidence', 0.0),
                                    position={'page': page_num, 'index': img_index},
                                    alt_text=processed_img.get('text', '')
                                )
                                
                                images.append(question_image)
                            
                            pix = None
                            
                        except Exception as e:
                            logger.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                            continue
            
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
            return images
    
    def _process_image_ocr(self, image_path: str) -> Dict[str, Any]:
        """Process image with OCR for text extraction"""
        if not config.ENABLE_OCR:
            return {'text': '', 'confidence': 0.0}
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return {'text': '', 'confidence': 0.0}
            
            # Enhance image for better OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # OCR with confidence
            ocr_data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
            text = ' '.join([word for word, conf in zip(ocr_data['text'], ocr_data['conf']) if conf > config.OCR_CONFIDENCE_THRESHOLD])
            
            avg_confidence = sum([conf for conf in ocr_data['conf'] if conf > 0]) / len([conf for conf in ocr_data['conf'] if conf > 0]) if ocr_data['conf'] else 0
            
            return {
                'text': text,
                'confidence': avg_confidence / 100.0,
                'ocr_data': ocr_data
            }
            
        except Exception as e:
            logger.warning(f"OCR processing failed for {image_path}: {str(e)}")
            return {'text': '', 'confidence': 0.0}
    
    def _classify_image_type(self, image_path: str) -> str:
        """Classify image type based on content"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 'unknown'
            
            # Simple heuristics for image classification
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect lines for diagrams/graphs
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) > 20:
                return 'diagram'
            elif lines is not None and len(lines) > 5:
                return 'graph'
            else:
                return 'equation'
                
        except Exception as e:
            logger.warning(f"Image classification failed: {str(e)}")
            return 'unknown'
    
    def _detect_questions_advanced(self, text_data: List[Dict[str, Any]], images: List[QuestionImage], test_id: str) -> List[Question]:
        """Advanced question detection with context awareness"""
        questions = []
        
        # Combine all text
        full_text = '\n'.join([page['text'] for page in text_data])
        
        # Detect question boundaries
        boundaries = self._detect_question_boundaries_advanced(full_text)
        
        for i, (start, end) in enumerate(boundaries):
            try:
                question_text = full_text[start:end].strip()
                
                if len(question_text) < 30:  # Skip very short segments
                    continue
                
                # Determine question type
                q_type, confidence = self._classify_question_type_advanced(question_text)
                
                # Extract options
                options = self._extract_options_advanced(question_text, q_type)
                
                # Detect subject and topic
                subject = self._detect_subject(question_text)
                topic = self._detect_topic(question_text, subject)
                
                # Find associated images
                question_images = self._associate_images_with_question(images, i, len(boundaries))
                
                # Create question
                question = Question(
                    id=f"{test_id}_q{i+1:03d}",
                    test_id=test_id,
                    type=q_type,
                    text=question_text,
                    options=options,
                    images=question_images,
                    section=self._determine_section(i, len(boundaries)),
                    subject=subject,
                    topic=topic,
                    difficulty=self._estimate_difficulty(question_text),
                    confidence=confidence,
                    page_number=self._find_page_number(start, text_data),
                    position={'start': start, 'end': end},
                    metadata={
                        'word_count': len(question_text.split()),
                        'has_math': self._has_mathematical_content(question_text),
                        'has_images': len(question_images) > 0
                    },
                    created_at=datetime.now()
                )
                
                questions.append(question)
                
            except Exception as e:
                logger.warning(f"Error processing question {i+1}: {str(e)}")
                continue
        
        return questions
    
    def _detect_question_boundaries_advanced(self, text: str) -> List[Tuple[int, int]]:
        """Advanced question boundary detection"""
        boundaries = []
        
        # Try multiple patterns
        for pattern_type in self.patterns['question_boundaries']:
            for pattern in self.patterns['question_boundaries'][pattern_type]:
                matches = list(re.finditer(pattern, text, re.MULTILINE))
                if matches:
                    for i, match in enumerate(matches):
                        start = match.start()
                        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                        boundaries.append((start, end))
                    break
            if boundaries:
                break
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        return boundaries
    
    def _classify_question_type_advanced(self, question_text: str) -> Tuple[str, float]:
        """Advanced question type classification"""
        text_lower = question_text.lower()
        
        # Initialize scores
        scores = {
            'SCQ': 0.0,
            'MCQ': 0.0,
            'INTEGER': 0.0,
            'MATCH_COLUMN': 0.0
        }
        
        # Check indicators for each type
        for q_type in ['scq', 'mcq', 'integer', 'match_column']:
            type_patterns = self.patterns['question_types'][q_type]
            
            for indicator in type_patterns.get('indicators', []):
                if re.search(indicator, question_text, re.IGNORECASE):
                    scores[q_type.upper()] += 0.4
            
            # Check option patterns
            if 'options_pattern' in type_patterns:
                options = re.findall(type_patterns['options_pattern'], question_text, re.DOTALL)
                expected_options = type_patterns.get('required_options', 0)
                
                if len(options) == expected_options:
                    scores[q_type.upper()] += 0.6
                elif len(options) > 0:
                    scores[q_type.upper()] += 0.2
            
            # Special handling for integer questions
            if q_type == 'integer' and type_patterns.get('no_options', False):
                option_count = len(re.findall(r'\([A-D]\)', question_text))
                if option_count == 0:
                    scores['INTEGER'] += 0.5
        
        # Determine best type
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type], 1.0)
        
        # Default to SCQ if confidence is too low
        if confidence < 0.3:
            best_type = 'SCQ'
            confidence = 0.3
        
        return best_type, confidence
    
    def _extract_options_advanced(self, question_text: str, question_type: str) -> List[str]:
        """Advanced option extraction with validation"""
        options = []
        
        if question_type in ['SCQ', 'MCQ']:
            # Extract standard options
            pattern = r'\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|\n\s*\d+[.)]|\Z)'
            matches = re.findall(pattern, question_text, re.DOTALL)
            
            for letter, text in matches:
                cleaned_text = re.sub(r'\s+', ' ', text.strip())
                options.append(f"({letter}) {cleaned_text}")
        
        elif question_type == 'MATCH_COLUMN':
            # Extract column data
            column_pattern = r'(?:Column\s+[I1]|List\s+[I1])(.*?)(?:Column\s+[I2]|List\s+[I2])(.*?)(?:\n\n|$)'
            match = re.search(column_pattern, question_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                col1, col2 = match.groups()
                
                # Extract items from each column
                item_pattern = r'\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|\n|$)'
                col1_items = re.findall(item_pattern, col1)
                col2_items = re.findall(r'(\d+)[.)]?\s*([^0-9]+?)(?=\d+[.)]|\n|$)', col2)
                
                for letter, text in col1_items:
                    options.append(f"Column I - ({letter}) {text.strip()}")
                
                for num, text in col2_items:
                    options.append(f"Column II - ({num}) {text.strip()}")
        
        return options
    
    def _detect_subject(self, question_text: str) -> str:
        """Detect subject based on content"""
        text_lower = question_text.lower()
        
        subject_scores = {'Physics': 0, 'Chemistry': 0, 'Mathematics': 0}
        
        for subject, keywords in self.patterns['subjects'].items():
            for keyword in keywords:
                if re.search(keyword, text_lower):
                    subject_scores[subject.capitalize()] += 1
        
        # Use ML classifier if available
        if self.ml_classifier:
            ml_subject = self.ml_classifier.predict_subject(question_text)
            if ml_subject:
                subject_scores[ml_subject] += 3
        
        return max(subject_scores, key=subject_scores.get) if max(subject_scores.values()) > 0 else 'General'
    
    def _detect_topic(self, question_text: str, subject: str) -> str:
        """Detect topic within subject"""
        # This would be enhanced with a more comprehensive topic database
        return 'General'
    
    def _estimate_difficulty(self, question_text: str) -> str:
        """Estimate question difficulty"""
        # Simple heuristics - can be enhanced with ML
        word_count = len(question_text.split())
        math_complexity = len(re.findall(r'[∫∑∏√±∞]', question_text))
        
        if word_count < 50 and math_complexity < 2:
            return 'Easy'
        elif word_count > 100 or math_complexity > 5:
            return 'Hard'
        else:
            return 'Medium'
    
    def _has_mathematical_content(self, text: str) -> bool:
        """Check if text contains mathematical content"""
        math_indicators = [
            r'[∫∑∏√±∞]', r'\^', r'_', r'\\[a-zA-Z]+',
            r'\d+\s*[+\-*/]\s*\d+', r'[a-zA-Z]\s*=\s*\d+'
        ]
        
        for pattern in math_indicators:
            if re.search(pattern, text):
                return True
        return False
    
    def _find_page_number(self, start_pos: int, text_data: List[Dict[str, Any]]) -> int:
        """Find page number for a given text position"""
        cumulative_length = 0
        for i, page in enumerate(text_data):
            cumulative_length += len(page['text'])
            if start_pos < cumulative_length:
                return i + 1
        return 1
    
    def _determine_section(self, question_index: int, total_questions: int) -> str:
        """Determine section based on question position"""
        # Simple equal division - can be enhanced
        section_size = total_questions // 3
        if question_index < section_size:
            return 'Section A'
        elif question_index < 2 * section_size:
            return 'Section B'
        else:
            return 'Section C'
    
    def _associate_images_with_question(self, images: List[QuestionImage], question_index: int, total_questions: int) -> List[QuestionImage]:
        """Associate images with questions based on position"""
        # Simple association - can be enhanced with spatial analysis
        images_per_question = len(images) // total_questions if total_questions > 0 else 0
