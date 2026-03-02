"""
Google Cloud Vision SDK — Multilingual OCR.
Uses DOCUMENT_TEXT_DETECTION with language hints for non-Latin scripts.
Requires GOOGLE_APPLICATION_CREDENTIALS set in the environment.
"""

import io
import re
from PIL import Image
from typing import Tuple, List
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("multilingual_ocr")

_MULTILINGUAL_HINTS = ["hi", "ur", "ar", "fa", "bn", "ta", "te", "mr", "gu", "pa", "en"]


def requires_multilingual_ocr(text: str) -> bool:
    """
    Check if text contains characters outside standard Latin script.

    Args:
        text: Sample text from primary OCR.

    Returns:
        True if non-Latin scripts detected.
    """
    latin_pattern = re.compile(r'^[A-Za-z0-9\s!@#$%^&*()_+\-=\[\]{}|\\:;"\'<>,.?/`~]*$')
    sample = text[:100]
    return not bool(latin_pattern.match(sample))


@retry_with_backoff(max_retries=3, base_delay=2.0, rate_limit=True)
def extract_multilingual_text(img: Image.Image) -> Tuple[str, float]:
    """
    Extract multilingual text using Google Cloud Vision SDK
    DOCUMENT_TEXT_DETECTION with language hints.

    Args:
        img: PIL Image containing text.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        vision_img = vision.Image(content=buf.getvalue())

        image_context = vision.ImageContext(language_hints=_MULTILINGUAL_HINTS)

        response = client.document_text_detection(
            image=vision_img,
            image_context=image_context
        )

        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return "", 0.0

        full_text = response.full_text_annotation.text.strip()

        # Collect detected languages and confidences from pages
        detected_langs: List[str] = []
        confidences = []
        for page in response.full_text_annotation.pages:
            for lang in page.property.detected_languages:
                if lang.language_code:
                    detected_langs.append(lang.language_code)
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        confidences.append(word.confidence)

        avg_conf = (sum(confidences) / len(confidences)) if confidences else (0.85 if full_text else 0.0)
        logger.debug(f"Multilingual OCR: {len(full_text)} chars, langs={detected_langs}, confidence={avg_conf:.3f}")

        return full_text, avg_conf

    except Exception as e:
        logger.error(f"Multilingual OCR failed: {e}")
        return "", 0.0
