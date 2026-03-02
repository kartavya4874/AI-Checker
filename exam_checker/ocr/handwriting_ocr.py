"""
Google Cloud Vision SDK — Handwritten text OCR.
Uses DOCUMENT_TEXT_DETECTION feature which handles handwriting and print.
Requires GOOGLE_APPLICATION_CREDENTIALS set in the environment.
"""

import io
from PIL import Image
from typing import Tuple
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("handwriting_ocr")


@retry_with_backoff(max_retries=3, base_delay=2.0, rate_limit=True)
def extract_handwritten_text(img: Image.Image) -> Tuple[str, float]:
    """
    Extract handwritten (and printed) text from image using
    Google Cloud Vision SDK DOCUMENT_TEXT_DETECTION.

    Args:
        img: PIL Image containing handwritten text.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        vision_img = vision.Image(content=buf.getvalue())

        response = client.document_text_detection(image=vision_img)

        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return "", 0.0

        full_text = response.full_text_annotation.text.strip()

        # Compute average word confidence from pages
        confidences = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        confidences.append(word.confidence)

        avg_conf = (sum(confidences) / len(confidences)) if confidences else (0.9 if full_text else 0.0)

        logger.debug(f"Handwriting OCR: {len(full_text)} chars, confidence={avg_conf:.3f}")
        return full_text, avg_conf

    except Exception as e:
        logger.error(f"Handwriting OCR failed: {e}")
        return "", 0.0
