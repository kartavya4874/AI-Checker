"""
Gemini API for printed text.
"""

from PIL import Image
from typing import Tuple
import google.generativeai as genai
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("print_ocr")


@retry_with_backoff(max_retries=3, base_delay=2.0, rate_limit=True)
def extract_printed_text(img: Image.Image) -> Tuple[str, float]:
    """
    Extract printed text from image using Gemini API.

    Args:
        img: PIL Image containing printed text.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = "Extract all the printed text from this image exactly as written. Preserve formatting where possible. If the image is completely blank or contains no text, output nothing."
        
        response = model.generate_content([prompt, img])
        
        text = response.text.strip()
        confidence = 0.95 if text else 0.0

        logger.debug(f"Print OCR: {len(text)} chars, confidence={confidence:.3f}")
        return text, confidence

    except Exception as e:
        logger.error(f"Print OCR failed: {e}")
        return "", 0.0
