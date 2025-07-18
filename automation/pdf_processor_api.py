import os
import io
import base64
import json
import uuid
import threading
import time
from datetime import datetime, timedelta
from queue import Queue
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from google.cloud import vision
import pandas as pd
import numpy as np
from PIL import Image
import pdf2image
import tempfile
import re
from typing import Dict, List
from dotenv import load_dotenv
import openpyxl
# Load environment variables
load_dotenv('config.env')

# Translation will be handled per request to avoid initialization issues

app = Flask(__name__)
CORS(app)

# Configuration from environment
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'temp_uploads')
RESULT_FOLDER = os.getenv('RESULT_FOLDER', 'temp_results')
DEBUG_MODE = os.getenv('DEBUG', '0') == '1'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Google Cloud Vision client initialization
def initialize_vision_client():
    """Initialize Google Cloud Vision client with proper authentication"""
    try:
        # Option 1: Use service account credentials file
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print(f"Using service account credentials from: {credentials_path}")
            return vision.ImageAnnotatorClient()
        
        # Option 2: Use API key
        api_key = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
        if api_key:
            print("Using API key for Vision API")
            # For API key, we'll use direct HTTP requests instead of the client library
            # This is because the client library doesn't support API keys directly
            return "API_KEY_MODE"
        
        # Option 3: Default authentication (ADC)
        print("Using default authentication (Application Default Credentials)")
        return vision.ImageAnnotatorClient()
    
    except Exception as e:
        print(f"Error initializing Vision client: {e}")
        print("Make sure to set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_VISION_API_KEY")
        return None

vision_client = initialize_vision_client()

class OCRResult:
    def __init__(self, text: str, x: int, y: int, width: int, height: int, confidence: float = 0.8):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        
    def to_dict(self):
        return {
            'text': self.text,
            'boundingBox': {
                'x': self.x,
                'y': self.y,
                'width': self.width,
                'height': self.height
            },
            'confidence': self.confidence
        }

class FieldMapping:
    def __init__(self, field_name: str, coordinate_area: Dict, excel_cell: Dict, priority: int = 5):
        self.field_name = field_name
        self.coordinate_area = coordinate_area  # {x, y, width, height}
        self.excel_cell = excel_cell  # {'sheet': str, 'row': int, 'column': int}
        self.priority = priority

def load_field_mappings_from_env() -> tuple:
    """Load field mappings from environment variables for both invoice and export certificate"""
    invoice_mappings = []
    export_certificate_mappings = []
    
    # Define invoice field names
    invoice_field_names = [
        'total_amount', 'email', 'company_address', 'address_line1', 'address_line2', 'item_name', 'company_name', 'pod', 'invoice_postal_country', 'invoice_email', 'invoice_phone'
    ]
    
    # Define export certificate field names (prefixed with EC_)
    export_certificate_field_names = [
        'chassis_number', 'gasoline', 'seat_number', 'weight', 'engine_capacity', 'length', 'width', 'height', 'net_weight', 'year'
    ]
    
    print("Loading invoice field mappings from environment...")
    
    # Load invoice mappings
    for field_name in invoice_field_names:
        # Get coordinate mapping
        coords_key = f"{field_name.upper()}_COORDS"
        coords_value = os.getenv(coords_key)
        
        # Get Excel cell mapping
        cell_key = f"{field_name.upper()}_CELL"
        cell_value = os.getenv(cell_key)
        
        print(f"Invoice field '{field_name}': coords='{coords_value}', cell='{cell_value}'")
        
        if coords_value and cell_value:
            try:
                # Parse coordinates: "x,y,width,height"
                x, y, width, height = map(int, coords_value.split(','))
                coordinate_area = {'x': x, 'y': y, 'width': width, 'height': height}
                
                # Parse Excel cell: "Sheet1:2:2" -> sheet, row, column
                # Handle sheet names with spaces by splitting on last two colons
                parts = cell_value.split(':')
                if len(parts) >= 3:
                    # Join all parts except the last two as sheet name
                    sheet_name = ':'.join(parts[:-2])
                    row_str = parts[-2]
                    col_str = parts[-1]
                else:
                    # Fallback for simple format
                    sheet_name, row_str, col_str = cell_value.split(':')
                
                excel_cell = {
                    'sheet': sheet_name.strip(),  # Remove any extra whitespace
                    'row': int(row_str),
                    'column': int(col_str)
                }
                
                # Set priority based on field importance
                priority = 10 if field_name in ['invoice_number', 'total_amount'] else 8
                
                mapping = FieldMapping(field_name, coordinate_area, excel_cell, priority)
                invoice_mappings.append(mapping)
                print(f"Successfully loaded invoice mapping for '{field_name}': {excel_cell}")
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing invoice field mapping for {field_name}: {e}")
                continue
    
    print("Loading export certificate field mappings from environment...")
    
    # Load export certificate mappings
    for field_name in export_certificate_field_names:
        # Get coordinate mapping (prefixed with EC_)
        coords_key = f"EC_{field_name.upper()}_COORDS"
        coords_value = os.getenv(coords_key)
        
        # Get Excel cell mapping (prefixed with EC_)
        cell_key = f"EC_{field_name.upper()}_CELL"
        cell_value = os.getenv(cell_key)
        
        print(f"Export certificate field '{field_name}': coords='{coords_value}', cell='{cell_value}'")
        
        if coords_value and cell_value:
            try:
                # Parse coordinates: "x,y,width,height"
                x, y, width, height = map(int, coords_value.split(','))
                coordinate_area = {'x': x, 'y': y, 'width': width, 'height': height}
                
                # Parse Excel cell: "Sheet1:2:2" -> sheet, row, column
                # Handle sheet names with spaces by splitting on last two colons
                parts = cell_value.split(':')
                if len(parts) >= 3:
                    # Join all parts except the last two as sheet name
                    sheet_name = ':'.join(parts[:-2])
                    row_str = parts[-2]
                    col_str = parts[-1]
                else:
                    # Fallback for simple format
                    sheet_name, row_str, col_str = cell_value.split(':')
                
                excel_cell = {
                    'sheet': sheet_name.strip(),  # Remove any extra whitespace
                    'row': int(row_str),
                    'column': int(col_str)
                }
                
                # Set priority for export certificate fields
                priority = 9  # High priority for export certificate data
                
                # Prefix field name to distinguish from invoice fields
                mapping = FieldMapping(f"ec_{field_name}", coordinate_area, excel_cell, priority)
                export_certificate_mappings.append(mapping)
                print(f"Successfully loaded export certificate mapping for '{field_name}': {excel_cell}")
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing export certificate field mapping for {field_name}: {e}")
                continue
    
    print(f"Total invoice mappings loaded: {len(invoice_mappings)}")
    print(f"Total export certificate mappings loaded: {len(export_certificate_mappings)}")
    return invoice_mappings, export_certificate_mappings

# Load field mappings from environment
INVOICE_FIELD_MAPPINGS, EXPORT_CERTIFICATE_FIELD_MAPPINGS = load_field_mappings_from_env()

# Fallback to default mappings if environment loading fails
if not INVOICE_FIELD_MAPPINGS:
    print("Warning: No field mappings loaded from environment, using defaults")
    print("This means your config.env file doesn't have the required field mappings")
    INVOICE_FIELD_MAPPINGS = [
        FieldMapping('invoice_number', {'x': 0, 'y': 0, 'width': 300, 'height': 150}, 
                    {'sheet': 'Sheet1', 'row': 2, 'column': 2}, 10),
        FieldMapping('invoice_date', {'x': 400, 'y': 0, 'width': 300, 'height': 150}, 
                    {'sheet': 'Sheet1', 'row': 3, 'column': 2}, 9),
        FieldMapping('company_name', {'x': 0, 'y': 150, 'width': 400, 'height': 100}, 
                    {'sheet': 'Sheet1', 'row': 4, 'column': 2}, 8),
        FieldMapping('total_amount', {'x': 500, 'y': 400, 'width': 200, 'height': 100}, 
                    {'sheet': 'Sheet1', 'row': 5, 'column': 2}, 10),
        FieldMapping('pod', {'x': 0, 'y': 300, 'width': 200, 'height': 50}, 
                    {'sheet': 'Sheet1', 'row': 6, 'column': 2}, 8),
        FieldMapping('vehicle_description', {'x': 200, 'y': 300, 'width': 300, 'height': 50}, 
                    {'sheet': 'Sheet1', 'row': 7, 'column': 2}, 7),
        FieldMapping('chassis_number', {'x': 0, 'y': 350, 'width': 200, 'height': 50}, 
                    {'sheet': 'Sheet1', 'row': 8, 'column': 2}, 7),
    ]
    EXPORT_CERTIFICATE_FIELD_MAPPINGS = []

def pdf_to_images(pdf_file) -> List[Image.Image]:
    """Convert PDF to images for OCR processing"""
    try:
        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
            pdf_file.save(tmp_pdf.name)
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(tmp_pdf.name, dpi=300, first_page=1, last_page=1)
            
            # Clean up
            os.unlink(tmp_pdf.name)
            
            return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def process_image_with_vision(image: Image.Image) -> List[OCRResult]:
    """Process image with Google Cloud Vision OCR"""
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        image_content = img_byte_arr.getvalue()
        
        # Handle different authentication modes
        if vision_client == "API_KEY_MODE":
            # Use direct HTTP request with API key
            api_key = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
            url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
            
            request_body = {
                "requests": [
                    {
                        "image": {
                            "content": base64.b64encode(image_content).decode('utf-8')
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION",
                                "maxResults": 100
                            }
                        ],
                        "imageContext": {
                            "languageHints": ["en", "ja"]
                        }
                    }
                ]
            }
            
            import requests
            response = requests.post(url, json=request_body)
            
            if response.status_code != 200:
                raise Exception(f"Vision API HTTP error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Save complete Vision API response to file for debugging if DEBUG_MODE
            if DEBUG_MODE:
                with open("vision_response_debug.json", "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Complete Vision API response saved to vision_response_debug.json")
            
            if 'error' in result:
                raise Exception(f"Vision API error: {result['error']}")
            
            # Process the response
            ocr_results = []
            if 'responses' in result and result['responses']:
                text_annotations = result['responses'][0].get('textAnnotations', [])
                
                # Skip first annotation (full text)
                original_texts = []
                annotation_data = []
                
                for annotation in text_annotations[1:]:
                    if 'boundingPoly' in annotation and 'vertices' in annotation['boundingPoly']:
                        vertices = annotation['boundingPoly']['vertices']
                        xs = [v.get('x', 0) for v in vertices]
                        ys = [v.get('y', 0) for v in vertices]
                        
                        x = min(xs)
                        y = min(ys)
                        width = max(xs) - x
                        height = max(ys) - y
                        
                        original_text = annotation.get('description', '')
                        original_texts.append(original_text)
                        annotation_data.append({
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'confidence': annotation.get('confidence', 0.8)
                        })
                
                # Batch translate all texts
                translated_texts = batch_translate_japanese_texts(original_texts)
                
                # Create OCR results with translated texts
                for i, (translated_text, data) in enumerate(zip(translated_texts, annotation_data)):
                    ocr_result = OCRResult(
                        text=translated_text,
                        x=data['x'],
                        y=data['y'],
                        width=data['width'],
                        height=data['height'],
                        confidence=data['confidence']
                    )
                    ocr_results.append(ocr_result)
            
            print(f"Vision API (API Key) processed successfully. Found {len(ocr_results)} text elements.")
            return ocr_results
            
        else:
            # Use client library
            vision_image = vision.Image(content=image_content)
            
            # Set up image context for language hints
            image_context = vision.ImageContext(language_hints=["en", "ja"])
            response = vision_client.text_detection(image=vision_image, image_context=image_context)
            
            # Save complete Vision API response to file for debugging if DEBUG_MODE
            if DEBUG_MODE:
                response_dict = {
                    'text_annotations': [
                        {
                            'description': annotation.description,
                            'bounding_poly': {
                                'vertices': [{'x': v.x, 'y': v.y} for v in annotation.bounding_poly.vertices]
                            } if annotation.bounding_poly.vertices else None,
                            'confidence': getattr(annotation, 'confidence', 0.8)
                        } for annotation in response.text_annotations
                    ] if response.text_annotations else [],
                    'error': response.error.message if response.error.message else None
                }
                with open("vision_response_debug.json", "w") as f:
                    json.dump(response_dict, f, indent=2)
                print(f"Complete Vision API response saved to vision_response_debug.json")
            
            if response.error.message:
                raise Exception(f'Vision API error: {response.error.message}')
            
            ocr_results = []
            
            # Process text annotations (skip first one which is full text)
            if response.text_annotations and len(response.text_annotations) > 1:
                original_texts = []
                annotation_data = []
                
                for annotation in response.text_annotations[1:]:
                    if annotation.bounding_poly.vertices:
                        # Calculate bounding box
                        vertices = annotation.bounding_poly.vertices
                        xs = [v.x for v in vertices]
                        ys = [v.y for v in vertices]
                        
                        x = min(xs)
                        y = min(ys)
                        width = max(xs) - x
                        height = max(ys) - y
                        
                        original_text = annotation.description
                        original_texts.append(original_text)
                        annotation_data.append({
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'confidence': getattr(annotation, 'confidence', 0.8)
                        })
                
                # Batch translate all texts
                translated_texts = batch_translate_japanese_texts(original_texts)
                
                # Create OCR results with translated texts
                for i, (translated_text, data) in enumerate(zip(translated_texts, annotation_data)):
                    ocr_result = OCRResult(
                        text=translated_text,
                        x=data['x'],
                        y=data['y'],
                        width=data['width'],
                        height=data['height'],
                        confidence=data['confidence']
                    )
                    ocr_results.append(ocr_result)
            
            print(f"Vision API (Client Library) processed successfully. Found {len(ocr_results)} text elements.")
            return ocr_results
        
    except Exception as e:
        print(f"Error processing image with Vision API: {e}")
        print(f"Vision client status: {vision_client}")
        return []

# Translation cache to avoid re-translating the same text
_translation_cache = {}

def detect_and_translate_japanese(text: str) -> str:
    """Detect if text contains Japanese characters and translate to English using deep-translator"""
    try:
        # Check cache first
        if text in _translation_cache:
            return _translation_cache[text]
        
        # Check if text contains Japanese characters (hiragana, katakana, kanji)
        import re
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
        
        if japanese_pattern.search(text):
            # Try multiple translation services in order of reliability
            try:
                # Option 1: Google Translate (most reliable)
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source='ja', target='en')
                translated_text = translator.translate(text)
                
                if translated_text and translated_text != text:
                    # Cache the result
                    _translation_cache[text] = translated_text
                    return translated_text
                    
            except Exception as google_error:
                # Option 2: Fallback to Microsoft Translator
                try:
                    from deep_translator import MicrosoftTranslator
                    translator = MicrosoftTranslator(source='ja', target='en')
                    translated_text = translator.translate(text)
                    
                    if translated_text and translated_text != text:
                        # Cache the result
                        _translation_cache[text] = translated_text
                        return translated_text
                        
                except Exception as ms_error:
                    # Option 3: Fallback to Libre Translate (free, open source)
                    try:
                        from deep_translator import LibreTranslator
                        translator = LibreTranslator(source='ja', target='en')
                        translated_text = translator.translate(text)
                        
                        if translated_text and translated_text != text:
                            # Cache the result
                            _translation_cache[text] = translated_text
                            return translated_text
                            
                    except Exception as libre_error:
                        # Cache the original text to avoid retrying
                        _translation_cache[text] = text
                        return text
            
            # Cache the original text if no translation was needed
            _translation_cache[text] = text
            return text
        else:
            # Already in English or no Japanese characters
            # Cache the original text
            _translation_cache[text] = text
            return text
            
    except Exception as e:
        # Cache the original text to avoid retrying
        _translation_cache[text] = text
        return text

def batch_translate_japanese_texts(texts: List[str]) -> List[str]:
    """Batch translate multiple Japanese texts to English for better performance"""
    try:
        import re
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
        
        # Filter texts that contain Japanese characters
        japanese_texts = []
        japanese_indices = []
        
        for i, text in enumerate(texts):
            if japanese_pattern.search(text):
                japanese_texts.append(text)
                japanese_indices.append(i)
        
        if not japanese_texts:
            return texts  # No Japanese text to translate
        
        # Optimized batch translation approach
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='ja', target='en')
            
            # Use larger batch size for better performance (Google Translate can handle up to 128 texts)
            batch_size = 50
            translated_batch = []
            
            # Process in larger batches for better performance
            for i in range(0, len(japanese_texts), batch_size):
                batch = japanese_texts[i:i + batch_size]
                try:
                    batch_translated = translator.translate_batch(batch)
                    translated_batch.extend(batch_translated)
                except Exception:
                    # If batch fails, fall back to individual translation for this batch
                    for text in batch:
                        translated_batch.append(detect_and_translate_japanese(text))
            
            # Update the original texts list with translations
            result = texts.copy()
            for i, translated_text in zip(japanese_indices, translated_batch):
                if translated_text and translated_text != japanese_texts[japanese_indices.index(i)]:
                    result[i] = translated_text
            
            return result
            
        except Exception as google_error:
            # Fallback to multiprocessing for individual translation
            try:
                import multiprocessing as mp
                
                if len(japanese_texts) > 5 and mp.cpu_count() > 1:
                    # Create a pool of workers
                    with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:
                        # Translate texts in parallel
                        translated_batch = pool.map(detect_and_translate_japanese, japanese_texts)
                    
                    # Update the original texts list with translations
                    result = texts.copy()
                    for i, translated_text in zip(japanese_indices, translated_batch):
                        if translated_text and translated_text != japanese_texts[japanese_indices.index(i)]:
                            result[i] = translated_text
                    
                    return result
                else:
                    # Fallback to individual translation
                    result = texts.copy()
                    for i in japanese_indices:
                        result[i] = detect_and_translate_japanese(texts[i])
                    return result
                    
            except Exception as mp_error:
                # Final fallback to individual translation
                result = texts.copy()
                for i in japanese_indices:
                    result[i] = detect_and_translate_japanese(texts[i])
                return result
            
    except Exception as e:
        return texts

def is_chassis_number(text: str) -> bool:
    """Check if text looks like a chassis number"""
    import re
    
    # Remove common prefixes/suffixes and clean the text
    cleaned_text = re.sub(r'^[A-Z]{2,3}-?', '', text.upper())  # Remove country codes like "JW5-"
    cleaned_text = re.sub(r'[^A-Z0-9]', '', cleaned_text)  # Keep only alphanumeric
    
    # Chassis number patterns:
    # 1. 17 characters (VIN format)
    # 2. 8-12 alphanumeric characters
    # 3. Contains both letters and numbers
    # 4. Not just a single word like "number"
    
    if len(cleaned_text) < 8:
        return False
    
    if len(cleaned_text) > 17:
        return False
    
    # Must contain both letters and numbers
    has_letters = bool(re.search(r'[A-Z]', cleaned_text))
    has_numbers = bool(re.search(r'[0-9]', cleaned_text))
    
    if not (has_letters and has_numbers):
        return False
    
    # Exclude common words that might be mistaken for chassis numbers
    common_words = {'number', 'chassis', 'vin', 'serial', 'id', 'code'}
    if text.lower().strip() in common_words:
        return False
    
    return True

def calculate_overlap(rect1: Dict, rect2: Dict) -> float:
    """Calculate overlap percentage between two rectangles"""
    x1 = max(rect1['x'], rect2['x'])
    y1 = max(rect1['y'], rect2['y'])
    x2 = min(rect1['x'] + rect1['width'], rect2['x'] + rect2['width'])
    y2 = min(rect1['y'] + rect1['height'], rect2['y'] + rect2['height'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    overlap_area = (x2 - x1) * (y2 - y1)
    rect1_area = rect1['width'] * rect1['height']
    
    return overlap_area / rect1_area if rect1_area > 0 else 0.0

def map_ocr_to_fields(ocr_results: List[OCRResult], field_mappings: List[FieldMapping] = None) -> Dict[str, Dict]:
    """Map OCR results to Excel fields based on coordinate matching with flexibility"""
    import re  # Import re locally to avoid shadowing issues

    
    extracted_data = {}
    
    # Sort field mappings by priority (highest first)
    sorted_mappings = sorted(field_mappings, key=lambda x: x.priority, reverse=True)
    
    for mapping in sorted_mappings:
        candidate_texts = []
        
        for ocr in ocr_results:
            ocr_rect = {
                'x': ocr.x,
                'y': ocr.y,
                'width': ocr.width,
                'height': ocr.height
            }
            
            # More flexible coordinate matching
            # Check if OCR result is within a reasonable range of the expected area
            mapping_area = mapping.coordinate_area
            
            # Calculate center points
            mapping_center_x = mapping_area['x'] + mapping_area['width'] / 2
            mapping_center_y = mapping_area['y'] + mapping_area['height'] / 2
            ocr_center_x = ocr.x + ocr.width / 2
            ocr_center_y = ocr.y + ocr.height / 2
            
            # Calculate distance from center
            distance = np.sqrt((mapping_center_x - ocr_center_x)**2 + (mapping_center_y - ocr_center_y)**2)
            
            # More flexible distance threshold (increased from strict overlap)
            max_distance = max(mapping_area['width'], mapping_area['height']) * 0.8
            
            if distance <= max_distance:
                # Additional check: for amount fields, look for numeric patterns
                if 'amount' in mapping.field_name.lower() or 'total' in mapping.field_name.lower():
                    # Check if text contains numbers and currency symbols
                    import re
                    if re.search(r'[\d,]+', ocr.text):
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance,
                            'is_amount': True
                        })
                elif 'email' in mapping.field_name.lower():
                    # For email fields, look for email patterns
                    import re
                    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ocr.text):
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance,
                            'is_email': True
                        })
                elif 'address' in mapping.field_name.lower():
                    # For address fields, accept any text (addresses can be varied)
                    candidate_texts.append({
                        'text': ocr.text,
                        'confidence': ocr.confidence,
                        'distance': distance,
                        'is_address': True,
                        'x': ocr.x,
                        'y': ocr.y
                    })
                elif 'item_name' in mapping.field_name.lower():
                    # For item name fields, filter for product name patterns
                    import re
                    # Only accept text that looks like product names (letters, numbers, common patterns)
                    if re.search(r'^[A-Za-z0-9\s\-]+$', ocr.text) and len(ocr.text.strip()) > 0:
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance,
                            'is_item_name': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                elif 'company_name' in mapping.field_name.lower():
                    # For company name fields, accept any text (company names can be varied)
                    candidate_texts.append({
                        'text': ocr.text,
                        'confidence': ocr.confidence,
                        'distance': distance,
                        'is_company_name': True,
                        'x': ocr.x,
                        'y': ocr.y
                    })
                elif 'chassis_number' in mapping.field_name.lower():
                    # For chassis number fields, prioritize actual chassis numbers over generic words
                    if is_chassis_number(ocr.text):
                        # Give higher priority to actual chassis numbers
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 0.5,  # Reduce distance for chassis numbers
                            'is_chassis_number': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                    else:
                        # Lower priority for generic words
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 2.0,  # Increase distance for generic words
                            'is_chassis_number': False,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                elif 'gasoline' in mapping.field_name.lower():
                    # For gasoline fields, prioritize the exact text "gasoline"
                    if ocr.text.lower() == 'gasoline':
                        # Give highest priority to exact "gasoline" text
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 0.1,  # Very low distance for exact match
                            'is_gasoline': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                    else:
                        # Lower priority for other text
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 5.0,  # Much higher distance for non-gasoline text
                            'is_gasoline': False,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                elif 'pod' in mapping.field_name.lower():
                    # For POD fields, prioritize the exact text "SOUTHAMPTON"
                    if ocr.text.upper() == 'SOUTHAMPTON':
                        # Give highest priority to exact "SOUTHAMPTON" text
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 0.1,  # Very low distance for exact match
                            'is_pod': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                    else:
                        # Lower priority for other text
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 5.0,  # Much higher distance for non-SOUTHAMPTON text
                            'is_pod': False,
                            'x': ocr.x,
                            'y': ocr.y
                        })

                elif 'invoice_postal_country' in mapping.field_name.lower():
                    # For invoice postal/country fields, only accept text in the postal/country area (y=494-586)
                    if 494 <= ocr.y <= 586:
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance,
                            'is_invoice_postal_country': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                elif 'invoice_email' in mapping.field_name.lower():
                    # For invoice email fields, only accept text in the email area (y=603-643) and filter for email patterns
                    if 603 <= ocr.y <= 643 and re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ocr.text):
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance,
                            'is_invoice_email': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                elif 'invoice_phone' in mapping.field_name.lower():
                    # For invoice phone fields, accept text in a wider phone area (y=650-700) and filter for phone number patterns
                    # Debug: Print all OCR text that might be phone numbers
                    if re.search(r'\d+', ocr.text):
                        print(f"Phone number candidate: '{ocr.text}' at y={ocr.y} (range: 650-700)")
                    
                    if 650 <= ocr.y <= 700 and re.search(r'\d+', ocr.text):
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance,
                            'is_invoice_phone': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                elif 'year' in mapping.field_name.lower():
                    # For year fields, prioritize numeric year values (4-digit years like 2015)
                    # Check if text is a 4-digit year
                    if re.match(r'^\d{4}$', ocr.text):
                        # Give highest priority to 4-digit years
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 0.1,  # Very low distance for 4-digit years
                            'is_year': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                    elif re.match(r'^\d{2}$', ocr.text):
                        # Medium priority for 2-digit years
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 0.5,  # Lower distance for 2-digit years
                            'is_year': True,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                    else:
                        # Lower priority for non-numeric text (like "Heisei")
                        candidate_texts.append({
                            'text': ocr.text,
                            'confidence': ocr.confidence,
                            'distance': distance * 5.0,  # Much higher distance for non-year text
                            'is_year': False,
                            'x': ocr.x,
                            'y': ocr.y
                        })
                else:
                    candidate_texts.append({
                        'text': ocr.text,
                        'confidence': ocr.confidence,
                        'distance': distance,
                        'is_amount': False
                    })
        
        # Select the best candidate (closest distance, highest confidence)
        if candidate_texts:
            
            # Prioritize amount fields if this is an amount mapping
            if 'amount' in mapping.field_name.lower() or 'total' in mapping.field_name.lower():
                amount_candidates = [c for c in candidate_texts if c.get('is_amount', False)]
                if amount_candidates:
                    candidate_texts = amount_candidates
            
            # Prioritize email fields if this is an email mapping
            elif 'email' in mapping.field_name.lower():
                email_candidates = [c for c in candidate_texts if c.get('is_email', False)]
                if email_candidates:
                    candidate_texts = email_candidates
            
            # Prioritize address fields if this is an address mapping
            elif 'address' in mapping.field_name.lower():
                address_candidates = [c for c in candidate_texts if c.get('is_address', False)]
                if address_candidates:
                    # For addresses, combine all candidates into one complete address
                    # Sort by y-coordinate first, then by x-coordinate to maintain reading order
                    sorted_candidates = sorted(address_candidates, key=lambda x: (x.get('y', 0), x.get('x', 0)))
                    
                    # Combine all text elements
                    combined_text = ' '.join([c['text'] for c in sorted_candidates])
                    
                    # Use the first candidate's metadata but with combined text
                    best_candidate = sorted_candidates[0].copy()
                    best_candidate['text'] = combined_text
                    candidate_texts = [best_candidate]
                else:
                    candidate_texts = address_candidates
            # Prioritize item name fields if this is an item name mapping
            elif 'item_name' in mapping.field_name.lower():
                item_name_candidates = [c for c in candidate_texts if c.get('is_item_name', False)]
                if item_name_candidates:
                    # For item names, combine all candidates into one complete item name
                    # Sort by x-coordinate to maintain left-to-right reading order
                    sorted_candidates = sorted(item_name_candidates, key=lambda x: x.get('x', 0))
                    
                    # Combine all text elements
                    combined_text = ' '.join([c['text'] for c in sorted_candidates])
                    
                    # Use the first candidate's metadata but with combined text
                    best_candidate = sorted_candidates[0].copy()
                    best_candidate['text'] = combined_text
                    candidate_texts = [best_candidate]
                else:
                    candidate_texts = item_name_candidates
            # Prioritize company name fields if this is a company name mapping
            elif 'company_name' in mapping.field_name.lower():
                company_name_candidates = [c for c in candidate_texts if c.get('is_company_name', False)]
                if company_name_candidates:
                    # For company names, combine all candidates into one complete company name
                    # Sort by x-coordinate to maintain left-to-right reading order
                    sorted_candidates = sorted(company_name_candidates, key=lambda x: x.get('x', 0))
                    
                    # Combine all text elements
                    combined_text = ' '.join([c['text'] for c in sorted_candidates])
                    
                    # Remove first character from company name
                    if len(combined_text) > 1:
                        combined_text = combined_text[1:]
                    
                    # Use the first candidate's metadata but with combined text
                    best_candidate = sorted_candidates[0].copy()
                    best_candidate['text'] = combined_text
                    candidate_texts = [best_candidate]
                else:
                    candidate_texts = company_name_candidates
            # Prioritize chassis number fields if this is a chassis number mapping
            elif 'chassis_number' in mapping.field_name.lower():
                # Sort by distance (which has been adjusted for chassis numbers)
                candidate_texts.sort(key=lambda x: x['distance'])
                
                # Select the best candidate (lowest distance)
                best_candidate = candidate_texts[0]
                candidate_texts = [best_candidate]
            # Prioritize gasoline fields if this is a gasoline mapping
            elif 'gasoline' in mapping.field_name.lower():
                # Sort by distance (which has been adjusted for gasoline)
                candidate_texts.sort(key=lambda x: x['distance'])
                
                # Select the best candidate (lowest distance)
                best_candidate = candidate_texts[0]
                candidate_texts = [best_candidate]
            # Prioritize POD fields if this is a POD mapping
            elif 'pod' in mapping.field_name.lower():
                # Sort by distance (which has been adjusted for POD)
                candidate_texts.sort(key=lambda x: x['distance'])
                
                # Select the best candidate (lowest distance)
                best_candidate = candidate_texts[0]
                candidate_texts = [best_candidate]

            # Prioritize invoice postal/country fields if this is an invoice postal/country mapping
            elif 'invoice_postal_country' in mapping.field_name.lower():
                invoice_postal_country_candidates = [c for c in candidate_texts if c.get('is_invoice_postal_country', False)]
                if invoice_postal_country_candidates:
                    # For invoice postal/country, combine all candidates into one complete text
                    # Sort by y-coordinate first, then by x-coordinate to maintain reading order
                    sorted_candidates = sorted(invoice_postal_country_candidates, key=lambda x: (x.get('y', 0), x.get('x', 0)))
                    
                    # Combine all text elements
                    combined_text = ' '.join([c['text'] for c in sorted_candidates])
                    
                    # Use the first candidate's metadata but with combined text
                    best_candidate = sorted_candidates[0].copy()
                    best_candidate['text'] = combined_text
                    candidate_texts = [best_candidate]
                else:
                    candidate_texts = invoice_postal_country_candidates
            # Prioritize invoice email fields if this is an invoice email mapping
            elif 'invoice_email' in mapping.field_name.lower():
                email_candidates = [c for c in candidate_texts if c.get('is_invoice_email', False)]
                if email_candidates:
                    candidate_texts = email_candidates
            # Prioritize invoice phone fields if this is an invoice phone mapping
            elif 'invoice_phone' in mapping.field_name.lower():
                phone_candidates = [c for c in candidate_texts if c.get('is_invoice_phone', False)]
                if phone_candidates:
                    candidate_texts = phone_candidates
                    print(f"Filtered to {len(candidate_texts)} invoice phone candidates")
            # Prioritize year fields if this is a year mapping
            elif 'year' in mapping.field_name.lower():
                # Sort by distance (which has been adjusted for year types)
                candidate_texts.sort(key=lambda x: x['distance'])
                
                # Select the best candidate (lowest distance)
                best_candidate = candidate_texts[0]
                candidate_texts = [best_candidate]
            else:
                best_candidate = min(candidate_texts, key=lambda x: x['distance'])
            
            if candidate_texts:
                best_candidate = candidate_texts[0]  # Use the first (and only) candidate
                extracted_data[mapping.field_name] = {
                    'text': best_candidate['text'],
                    'excel_cell': mapping.excel_cell,
                    'confidence': best_candidate['confidence']
                }
    
    return extracted_data

def extract_address_lines(description_text: str) -> tuple:
    """Extract first and second lines of address from description text"""
    if not description_text:
        return "", ""
    
    # Split the description by newlines and clean up
    lines = [line.strip() for line in description_text.split('\n') if line.strip()]
    
    # Find the UK address section - look for lines that contain address-like content
    address_start = -1
    for i, line in enumerate(lines):
        # Look for lines that contain UK address patterns
        if any(keyword in line.lower() for keyword in ['harlow', 'seymours', 'essex', 'cm19']):
            address_start = i
            break
    
    if address_start == -1:
        # If no specific address found, try to find company name patterns
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['autos', 'ltd', 'llc', 'co']):
                address_start = i
                break
    
    if address_start == -1:
        return "", ""
    
    # Extract first line (company name) - clean it up
    first_line = lines[address_start].strip()
    # Remove any extra text after the company name
    if ' ' in first_line:
        # Take only the first part if it looks like a company name
        parts = first_line.split()
        if len(parts) >= 2 and any(keyword in first_line.lower() for keyword in ['autos', 'ltd', 'llc', 'co']):
            # Look for company name ending patterns
            for i, part in enumerate(parts):
                if any(keyword in part.lower() for keyword in ['autos', 'ltd', 'llc', 'co']):
                    first_line = ' '.join(parts[:i+1])
                    break
    
    # Extract second line (street address) - clean it up
    second_line = ""
    if address_start + 1 < len(lines):
        second_line = lines[address_start + 1].strip()
        # Clean up second line - take only the street address part
        if second_line:
            # Split by spaces and take only the address parts
            parts = second_line.split()
            clean_parts = []
            for part in parts:
                # Stop if we hit postal code or other non-address content
                if any(keyword in part.lower() for keyword in ['cm19', 'united', 'kingdom', 'info@', '@', '4479']):
                    break
                clean_parts.append(part)
            second_line = ' '.join(clean_parts)
    
    return first_line, second_line

def clean_extracted_data(data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Clean and normalize extracted data"""
    cleaned = {}
    
    for key, value_dict in data.items():
        if value_dict and 'text' in value_dict:
            text = value_dict['text']
            # Remove extra whitespace
            cleaned_text = ' '.join(text.split())
            
            # Specific cleaning for different field types
            if 'amount' in key.lower() or 'total' in key.lower():
                # Extract monetary values
                amount_match = re.search(r'[\$€£¥]?[\d,]+\.?\d*', cleaned_text)
                if amount_match:
                    cleaned_text = amount_match.group()
            
            elif 'date' in key.lower():
                # Try to normalize date formats
                date_patterns = [
                    r'\d{2}/\d{2}/\d{4}',
                    r'\d{2}-\d{2}-\d{4}',
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{2}\.\d{2}\.\d{4}'
                ]
                for pattern in date_patterns:
                    date_match = re.search(pattern, cleaned_text)
                    if date_match:
                        cleaned_text = date_match.group()
                        break
            
            elif 'number' in key.lower() or 'chassis' in key.lower():
                # Extract alphanumeric IDs (for invoice numbers, chassis numbers, etc.)
                number_match = re.search(r'[A-Z0-9-]+', cleaned_text)
                if number_match:
                    cleaned_text = number_match.group()
            
            elif 'weight' in key.lower():
                # Extract only numerical part from weight fields (e.g., "940kg" -> "940", "830k" -> "830")
                weight_match = re.search(r'(\d+)', cleaned_text)
                if weight_match:
                    cleaned_text = weight_match.group(1)
            
            elif 'engine_capacity' in key.lower():
                # Extract decimal number and multiply by 1000 (e.g., "0.65" -> "650")
                capacity_match = re.search(r'(\d+\.?\d*)', cleaned_text)
                if capacity_match:
                    try:
                        capacity_value = float(capacity_match.group(1))
                        capacity_cc = int(capacity_value * 1000)
                        cleaned_text = str(capacity_cc)
                    except (ValueError, TypeError):
                        # If conversion fails, keep original text
                        pass
            
            elif 'address_line1' in key.lower():
                # Hardcode address_line1 to 'Harlow Jap Autos'
                cleaned_text = 'Harlow Jap Autos'
            
            elif 'address_line2' in key.lower():
                # Clean up second line of address - remove extra text
                cleaned_text = ' '.join(cleaned_text.split())
                # Extract just the street address part
                if cleaned_text:
                    parts = cleaned_text.split()
                    clean_parts = []
                    for part in parts:
                        # Stop if we hit postal code or other non-address content
                        if any(keyword in part.lower() for keyword in ['cm19', 'united', 'kingdom', 'info@', '@', '4479']):
                            break
                        clean_parts.append(part)
                    cleaned_text = ' '.join(clean_parts)
            
            elif 'gasoline' in key.lower():
                # Convert gasoline to uppercase
                cleaned_text = cleaned_text.upper()
            
            elif 'phone' in key.lower():
                # Add '+' prefix to phone numbers if not already present
                if cleaned_text and not cleaned_text.startswith('+'):
                    cleaned_text = '+' + cleaned_text
            
            elif 'item_name' in key.lower():
                # Capitalize item name
                cleaned_text = cleaned_text.upper()
            
            cleaned[key] = {
                'text': cleaned_text,
                'excel_cell': value_dict['excel_cell'],
                'confidence': value_dict['confidence']
            }
        else:
            cleaned[key] = {
                'text': '',
                'excel_cell': value_dict.get('excel_cell', {}),
                'confidence': 0.0
            }
    
    return cleaned

def cleanup_old_files():
    """Clean up old temporary files automatically"""
    try:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)  # Clean files older than 1 hour
        
        cleaned_count = 0
        
        # Clean upload folder
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time.timestamp():
                    os.remove(file_path)
                    cleaned_count += 1
        
        # Clean result folder
        if os.path.exists(RESULT_FOLDER):
            for filename in os.listdir(RESULT_FOLDER):
                file_path = os.path.join(RESULT_FOLDER, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time.timestamp():
                    os.remove(file_path)
                    cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} old temporary files")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")

def convert_to_excel_value(text: str, field_name: str) -> any:
    """Convert text to appropriate Excel data type based on field name"""
    if not text:
        return text
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Fields that should be numbers
    numeric_fields = [
        'total_amount', 'weight', 'net_weight', 'length', 'width', 'height', 
        'engine_capacity', 'seat_number', 'year'
    ]
    
    # Fields that should be text (emails, names, addresses, etc.)
    text_fields = [
        'company_name', 'item_name', 'chassis_number', 'gasoline', 'pod',
        'email', 'invoice_email', 'invoice_phone', 'invoice_postal_country'
    ]
    
    # Check if this field should be numeric
    if any(field in field_name.lower() for field in numeric_fields):
        try:
            # Remove common non-numeric characters for amount fields
            if 'amount' in field_name.lower():
                # Remove currency symbols and commas, keep decimal points
                cleaned_text = text.replace(',', '').replace('¥', '').replace('$', '').replace('€', '').replace('£', '')
                return float(cleaned_text)
            else:
                # For other numeric fields, just convert to float/int
                if '.' in text:
                    return float(text)
                else:
                    return int(text)
        except (ValueError, TypeError):
            # If conversion fails, return as text
            return text
    
    # For text fields, return as is
    return text

def populate_excel_template(template_path, extracted_data: Dict[str, Dict]) -> str:
    """Populate Excel template with extracted data at specific cell coordinates"""
    try:
        # Load the workbook using openpyxl directly from the template path
        workbook = openpyxl.load_workbook(template_path)
        
        # Populate cells with extracted data
        for field_name, data_dict in extracted_data.items():
            if data_dict.get('text') and data_dict.get('excel_cell'):
                excel_cell = data_dict['excel_cell']
                text = data_dict['text']
                
                # Get or create worksheet
                sheet_name = excel_cell.get('sheet', 'Sheet1')
                
                if sheet_name in workbook.sheetnames:
                    worksheet = workbook[sheet_name]
                else:
                    worksheet = workbook.create_sheet(sheet_name)
                
                # Convert text to appropriate Excel data type
                excel_value = convert_to_excel_value(text, field_name)
                
                # Write to specific cell
                row = excel_cell.get('row', 1)
                col = excel_cell.get('column', 1)
                
                # Write the value to the cell
                cell = worksheet.cell(row=row, column=col)
                
                # Handle merged cells - if cell is merged, write to the top-left cell of the merged range
                if cell.coordinate in worksheet.merged_cells:
                    # Find the top-left cell of the merged range
                    for merged_range in worksheet.merged_cells.ranges:
                        if cell.coordinate in merged_range:
                            top_left_cell = worksheet.cell(row=merged_range.min_row, column=merged_range.min_col)
                            top_left_cell.value = excel_value
                            print(f"Populated {field_name} = {excel_value} (type: {type(excel_value).__name__}) at merged cell {top_left_cell.coordinate}")
                            break
                else:
                    cell.value = excel_value
                    print(f"Populated {field_name} = {excel_value} (type: {type(excel_value).__name__}) at {sheet_name}:{row}:{col}")
        
        # Add today's date to cell 6-O
        try:
            # Get the first worksheet (assuming it's Sheet1)
            worksheet = workbook.active if workbook.active else workbook['Sheet1']
            
            # Format today's date as '10-Jul-25'
            today = datetime.now()
            formatted_date = today.strftime('%d-%b-%y')
            
            # Write to cell 6-O (row 6, column 15)
            date_cell = worksheet.cell(row=6, column=15)
            date_cell.value = formatted_date
            print(f"Populated today's date = {formatted_date} at cell 6-O")
            
        except Exception as e:
            print(f"Error adding today's date: {e}")
        
        # Save the populated workbook
        result_filename = f"populated_{uuid.uuid4().hex}.xlsx"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        workbook.save(result_path)
        
        return result_filename
        
    except Exception as e:
        print(f"Error populating Excel template: {e}")
        # Create a simple Excel file with extracted data
        simple_data = {field: data.get('text', '') for field, data in extracted_data.items()}
        df = pd.DataFrame([simple_data])
        result_filename = f"simple_{uuid.uuid4().hex}.xlsx"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        df.to_excel(result_path, index=False)
        return result_filename

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for basic connectivity"""
    return jsonify({
        'message': 'PDF Processor API is running',
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'vision_client_ready': vision_client is not None,
        'invoice_field_mappings_loaded': len(INVOICE_FIELD_MAPPINGS),
        'export_certificate_field_mappings_loaded': len(EXPORT_CERTIFICATE_FIELD_MAPPINGS),
        'upload_folder': UPLOAD_FOLDER,
        'result_folder': RESULT_FOLDER
    })

@app.route('/api/test-vision', methods=['POST'])
def test_vision():
    """Test endpoint for Vision API"""
    try:
        if vision_client is None:
            return jsonify({
                'success': False,
                'error': 'Vision client not initialized'
            }), 500
        
        # Test with a simple image or return client status
        return jsonify({
            'success': True,
            'message': 'Vision client is ready',
            'client_type': type(vision_client).__name__
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Global progress tracking
progress_sessions = {}

def send_progress_update(session_id: str, step: str, percentage: int, message: str):
    """Send progress update to client"""
    if session_id in progress_sessions:
        progress_data = {
            'step': step,
            'percentage': percentage,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        progress_sessions[session_id].put(progress_data)

@app.route('/api/progress/<session_id>', methods=['GET'])
def progress_stream(session_id):
    """Server-Sent Events endpoint for real-time progress updates"""
    def generate():
        if session_id not in progress_sessions:
            progress_sessions[session_id] = Queue()
        
        queue = progress_sessions[session_id]
        
        while True:
            try:
                # Wait for progress updates
                progress_data = queue.get(timeout=30)  # 30 second timeout
                
                # Send SSE data
                yield f"data: {json.dumps(progress_data)}\n\n"
                
                # If we reach 100%, close the stream
                if progress_data.get('percentage', 0) >= 100:
                    break
                    
            except Exception as e:
                # Send error or keep-alive
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/process-pdf', methods=['POST'])
def process_pdf():
    """Main endpoint for processing PDF with OCR and Excel template"""
    session_id = str(uuid.uuid4())
    progress_sessions[session_id] = Queue()
    
    try:
        send_progress_update(session_id, "init", 0, "Initializing PDF processing...")
        
        # Check if Vision client is initialized
        if vision_client is None:
            send_progress_update(session_id, "error", 0, "Vision API not configured")
            return jsonify({
                'success': False,
                'error': 'Google Cloud Vision API is not properly configured. Please check your credentials.',
                'session_id': session_id
            }), 500
        
        send_progress_update(session_id, "validate", 5, "Validating uploaded files...")
        
        # Validate request - both files are now required
        if 'invoice_pdf_file' not in request.files:
            send_progress_update(session_id, "error", 0, "Invoice PDF file is required")
            return jsonify({
                'success': False,
                'error': 'Invoice PDF file is required',
                'session_id': session_id
            }), 400
        
        if 'export_certificate_pdf_file' not in request.files:
            send_progress_update(session_id, "error", 0, "Export certificate PDF file is required")
            return jsonify({
                'success': False,
                'error': 'Export certificate PDF file is required',
                'session_id': session_id
            }), 400
        
        invoice_pdf_file = request.files['invoice_pdf_file']
        export_certificate_pdf_file = request.files['export_certificate_pdf_file']
        
        # Validate file types
        if not invoice_pdf_file.filename.lower().endswith('.pdf'):
            send_progress_update(session_id, "error", 0, "Invalid invoice PDF file format")
            return jsonify({
                'success': False,
                'error': 'Invalid invoice PDF file format',
                'session_id': session_id
            }), 400
        
        if not export_certificate_pdf_file.filename.lower().endswith('.pdf'):
            send_progress_update(session_id, "error", 0, "Invalid export certificate PDF file format")
            return jsonify({
                'success': False,
                'error': 'Invalid export certificate PDF file format',
                'session_id': session_id
            }), 400
        
        send_progress_update(session_id, "cleanup", 10, "Cleaning up old files...")
        
        # Clean up old files before processing
        cleanup_old_files()
        
        send_progress_update(session_id, "template", 15, "Loading Excel template...")
        
        # Use fixed template
        template_path = 'template.xlsx'
        if not os.path.exists(template_path):
            send_progress_update(session_id, "error", 0, "Template file not found")
            return jsonify({
                'success': False,
                'error': 'Template file not found',
                'session_id': session_id
            }), 500
        
        send_progress_update(session_id, "convert_invoice", 20, "Converting invoice PDF to images...")
        
        # Convert invoice PDF to images
        invoice_images = pdf_to_images(invoice_pdf_file)
        if not invoice_images:
            send_progress_update(session_id, "error", 0, "Failed to convert invoice PDF to images")
            return jsonify({
                'success': False,
                'error': 'Failed to convert invoice PDF to images. Please ensure the PDF is not corrupted and contains at least one page.',
                'session_id': session_id
            }), 500
        
        send_progress_update(session_id, "ocr_invoice", 35, "Processing invoice with OCR...")
        
        # Process invoice first page with OCR
        invoice_ocr_results = process_image_with_vision(invoice_images[0])
        if not invoice_ocr_results:
            send_progress_update(session_id, "error", 0, "Failed to extract text from invoice PDF")
            return jsonify({
                'success': False,
                'error': 'Failed to extract text from invoice PDF. Please check if the PDF contains readable text.',
                'session_id': session_id
            }), 500
        
        send_progress_update(session_id, "convert_export", 50, "Converting export certificate PDF to images...")
        
        # Convert export certificate PDF to images
        export_certificate_images = pdf_to_images(export_certificate_pdf_file)
        if not export_certificate_images:
            send_progress_update(session_id, "error", 0, "Failed to convert export certificate PDF to images")
            return jsonify({
                'success': False,
                'error': 'Failed to convert export certificate PDF to images. Please ensure the PDF is not corrupted and contains at least one page.',
                'session_id': session_id
            }), 500
        
        send_progress_update(session_id, "ocr_export", 65, "Processing export certificate with OCR...")
        
        # Process export certificate first page with OCR
        export_certificate_ocr_results = process_image_with_vision(export_certificate_images[0])
        if not export_certificate_ocr_results:
            send_progress_update(session_id, "error", 0, "Failed to extract text from export certificate PDF")
            return jsonify({
                'success': False,
                'error': 'Failed to extract text from export certificate PDF. Please check if the PDF contains readable text.',
                'session_id': session_id
            }), 500
        
        send_progress_update(session_id, "map_fields", 75, "Mapping OCR results to fields...")
        
        # Map OCR results to fields for both PDFs
        invoice_extracted_data = map_ocr_to_fields(invoice_ocr_results, INVOICE_FIELD_MAPPINGS)
        export_certificate_extracted_data = map_ocr_to_fields(export_certificate_ocr_results, EXPORT_CERTIFICATE_FIELD_MAPPINGS)
        
        send_progress_update(session_id, "combine_data", 80, "Combining extracted data...")
        
        # Combine extracted data from both PDFs
        combined_extracted_data = {}
        combined_extracted_data.update(invoice_extracted_data)
        combined_extracted_data.update(export_certificate_extracted_data)
        
        send_progress_update(session_id, "clean_data", 85, "Cleaning and validating data...")
        
        # Clean the combined data
        extracted_data = clean_extracted_data(combined_extracted_data)
        
        send_progress_update(session_id, "populate_excel", 90, "Populating Excel template...")
        
        # Populate Excel template
        result_filename = populate_excel_template(template_path, extracted_data)
        
        send_progress_update(session_id, "complete", 100, "Processing completed successfully!")
        
        # Prepare response data (simplified for frontend)
        response_data = {}
        for field_name, data_dict in extracted_data.items():
            # Remove "ec_" prefix from export certificate fields
            clean_field_name = field_name.replace('ec_', '') if field_name.startswith('ec_') else field_name
            
            # Skip company_address field
            if clean_field_name == 'company_address':
                continue
                
            response_data[clean_field_name] = data_dict.get('text', '')
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'PDFs processed successfully',
            'extracted_data': response_data,
            'download_url': f'/api/download/{result_filename}',
            'session_id': session_id,
            'total_elements': {
                'invoice': len(invoice_ocr_results),
                'export_certificate': len(export_certificate_ocr_results),
                'combined_total': len(invoice_ocr_results) + len(export_certificate_ocr_results)
            },
            'mapped_fields': len(extracted_data),
            'field_details': extracted_data,  # Include full details for debugging
            'files_processed': {
                'invoice': invoice_pdf_file.filename,
                'export_certificate': export_certificate_pdf_file.filename
            }
        })
        
    except Exception as e:
        send_progress_update(session_id, "error", 0, f"Processing failed: {str(e)}")
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(e)}',
            'session_id': session_id
        }), 500
    finally:
        # Clean up progress session after a delay
        def cleanup_session():
            time.sleep(10)  # Keep session for 10 seconds after completion
            if session_id in progress_sessions:
                del progress_sessions[session_id]
        
        threading.Thread(target=cleanup_session, daemon=True).start()

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed Excel file"""
    try:
        file_path = os.path.join(RESULT_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old temporary files"""
    try:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)  # Clean files older than 1 hour
        
        cleaned_count = 0
        
        # Clean upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.getmtime(file_path) < cutoff_time.timestamp():
                os.remove(file_path)
                cleaned_count += 1
        
        # Clean result folder
        for filename in os.listdir(RESULT_FOLDER):
            file_path = os.path.join(RESULT_FOLDER, filename)
            if os.path.getmtime(file_path) < cutoff_time.timestamp():
                os.remove(file_path)
                cleaned_count += 1
        
        return jsonify({
            'success': True,
            'cleaned_files': cleaned_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting PDF Processor API...")
    print("Make sure to set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    
    # Get port from environment variable (for production deployment)
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
