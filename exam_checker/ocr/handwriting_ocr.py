"""
Google Cloud Vision API v1 — Handwritten text OCR.
Uses the DOCUMENT_TEXT_DETECTION feature which handles both handwriting
and dense printed text via a simple HTTPS REST call (no SDK needed).
"""

import io
import base64
import requests
from PIL import Image
from typing import Tuple
from config import config
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("handwriting_ocr")

_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"


def _pil_to_b64(img: Image.Image) -> str:
    """Convert PIL image to base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@retry_with_backoff(max_retries=3, base_delay=2.0, rate_limit=True)
def extract_handwritten_text(img: Image.Image) -> Tuple[str, float]:
    """
    Extract handwritten (and printed) text from image using
    Google Cloud Vision API v1 DOCUMENT_TEXT_DETECTION.

    Args:
        img: PIL Image containing handwritten text.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    try:
        payload = {
            "requests": [
                {
                    "image": {"content": _pil_to_b64(img)},
                    "features": [
                        {"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}
                    ],
                }
            ]
        }

        resp = requests.post(
            _VISION_URL,
            params={"key": config.GOOGLE_VISION_API_KEY},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        response = data.get("responses", [{}])[0]

        # Check for API-level error
        if "error" in response:
            logger.error(f"Vision API error: {response['error']}")
            return "", 0.0

        full_text = response.get("fullTextAnnotation", {}).get("text", "").strip()

        # Compute average word confidence from pages
        confidences = []
        for page in response.get("fullTextAnnotation", {}).get("pages", []):
            for block in page.get("blocks", []):
                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        c = word.get("confidence", None)
                        if c is not None:
                            confidences.append(c)

        avg_conf = (sum(confidences) / len(confidences)) if confidences else (0.9 if full_text else 0.0)

        logger.debug(f"Handwriting OCR: {len(full_text)} chars, confidence={avg_conf:.3f}")
        return full_text, avg_conf

    except Exception as e:
        logger.error(f"Handwriting OCR failed: {e}")
        return "", 0.0
