"""
Google Cloud Vision API v1 — Multilingual OCR.
Uses DOCUMENT_TEXT_DETECTION with language hints for non-Latin scripts
(Hindi, Urdu, Arabic, etc.).
"""

import io
import re
import base64
import requests
from PIL import Image
from typing import Tuple, List
from config import config
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("multilingual_ocr")

_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"

# BCP-47 language hints passed to Vision API for non-Latin content
_MULTILINGUAL_HINTS = ["hi", "ur", "ar", "fa", "bn", "ta", "te", "mr", "gu", "pa", "en"]


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
    Extract multilingual text (Hindi, Urdu, Arabic, etc.) using
    Google Cloud Vision API v1 DOCUMENT_TEXT_DETECTION with language hints.

    Args:
        img: PIL Image containing text.

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
                    "imageContext": {
                        "languageHints": _MULTILINGUAL_HINTS
                    },
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

        full_text = response.get("fullTextAnnotation", {}).get("text", "").strip()

        # Collect detected languages from pages
        detected_langs: List[str] = []
        confidences = []
        for page in response.get("fullTextAnnotation", {}).get("pages", []):
            for lang in page.get("property", {}).get("detectedLanguages", []):
                code = lang.get("languageCode", "")
                if code:
                    detected_langs.append(code)
            for block in page.get("blocks", []):
                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        c = word.get("confidence", None)
                        if c is not None:
                            confidences.append(c)

        avg_conf = (sum(confidences) / len(confidences)) if confidences else (0.85 if full_text else 0.0)
        logger.debug(f"Multilingual OCR: {len(full_text)} chars, langs={detected_langs}, confidence={avg_conf:.3f}")

        return full_text, avg_conf

    except Exception as e:
        logger.error(f"Multilingual OCR failed: {e}")
        return "", 0.0
