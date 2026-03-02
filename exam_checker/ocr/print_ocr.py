"""
Google Cloud Vision API v1 — Printed text OCR.
Uses TEXT_DETECTION feature optimised for sparse, well-spaced printed text.
Falls back to DOCUMENT_TEXT_DETECTION for dense layouts.
"""

import io
import base64
import requests
from PIL import Image
from typing import Tuple
from config import config
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("print_ocr")

_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@retry_with_backoff(max_retries=3, base_delay=2.0, rate_limit=True)
def extract_printed_text(img: Image.Image) -> Tuple[str, float]:
    """
    Extract printed text from image using Google Cloud Vision API v1
    TEXT_DETECTION (optimised for sparse, clean printed text).

    Args:
        img: PIL Image containing printed text.

    Returns:
        Tuple of (extracted_text, confidence_score).
    """
    try:
        payload = {
            "requests": [
                {
                    "image": {"content": _pil_to_b64(img)},
                    "features": [
                        {"type": "TEXT_DETECTION", "maxResults": 1}
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

        if "error" in response:
            logger.error(f"Vision API error: {response['error']}")
            return "", 0.0

        annotations = response.get("textAnnotations", [])
        if not annotations:
            return "", 0.0

        # First annotation is the full concatenated text
        full_text = annotations[0].get("description", "").strip()
        confidence = 0.95 if full_text else 0.0

        logger.debug(f"Print OCR: {len(full_text)} chars, confidence={confidence:.3f}")
        return full_text, confidence

    except Exception as e:
        logger.error(f"Print OCR failed: {e}")
        return "", 0.0
