"""
Gemini API for multilingual text.
"""

import re
from PIL import Image
from typing import Tuple
import google.generativeai as genai
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("multilingual_ocr")


def requires_multilingual_ocr(text: str) -> bool:
    """
    Check if text contains characters outside standard Latin script.
    
    Args:
        text: Sample text from primary OCR.
        
    Returns:
        True if non-Latin scripts are detected.
    """
    # Regex for standard Latin, digits, punctuation, and common symbols
    # If there are characters NOT in this set, we might need multilingual OCR
    latin_pattern = re.compile(r'^[A-Za-z0-9\s!@#$%^&*()_+\-=\[\]{}|\\:;"\'<>,.?/`~]*$')
    
    # Check a sample of the text (up to 100 chars) for performance
    sample = text[:100]
    return not bool(latin_pattern.match(sample))


@retry_with_backoff(max_retries=3, base_delay=2.0, rate_limit=True)
def extract_multilingual_text(img: Image.Image) -> Tuple[str, float]:
    """
    Extract multilingual text (Hindi, Urdu, Arabic, etc.) using Gemini API.

    Args:
        img: PIL Image containing text.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = "Extract all the text from this image exactly as written. The image may contain multiple languages, including non-Latin scripts like Hindi, Urdu, or Arabic. Preserve formatting and output exactly what is in the image."
        
        response = model.generate_content([prompt, img])
        
        text = response.text.strip()
        confidence = 0.85 if text else 0.0

        logger.debug(f"Multilingual OCR: {len(text)} chars, confidence={confidence:.3f}")
        return text, confidence

    except Exception as e:
        logger.error(f"Multilingual OCR failed: {e}")
        return "", 0.0
