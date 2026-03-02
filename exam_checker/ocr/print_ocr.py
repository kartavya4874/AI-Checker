"""
Google Cloud Vision SDK — Printed text OCR.
Uses TEXT_DETECTION feature optimized for sparse, well-spaced printed text.
Requires GOOGLE_APPLICATION_CREDENTIALS set in the environment.
"""

import io
from PIL import Image
from typing import Tuple
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("print_ocr")


@retry_with_backoff(max_retries=3, base_delay=2.0, rate_limit=True)
def extract_printed_text(img: Image.Image) -> Tuple[str, float]:
    """
    Extract printed text from image using Google Cloud Vision SDK
    TEXT_DETECTION.

    Args:
        img: PIL Image containing printed text.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        vision_img = vision.Image(content=buf.getvalue())

        response = client.text_detection(image=vision_img)

        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return "", 0.0

        annotations = response.text_annotations
        if not annotations:
            return "", 0.0

        # First annotation is the full concatenated text
        full_text = annotations[0].description.strip()
        confidence = 0.95 if full_text else 0.0

        logger.debug(f"Print OCR: {len(full_text)} chars, confidence={confidence:.3f}")
        return full_text, confidence

    except Exception as e:
        logger.error(f"Print OCR failed: {e}")
        return "", 0.0
