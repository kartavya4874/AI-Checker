"""
OCR Router — routes image to best OCR result.
Blank check first, then Vision API, language detection for multilingual.
"""

from PIL import Image
from typing import Dict, Any
from preprocessing.blank_detector import is_blank
from ocr.handwriting_ocr import extract_handwritten_text
from ocr.multilingual_ocr import extract_multilingual_text, has_non_latin
from utils.logger import get_logger

logger = get_logger("ocr_router")


def route_ocr(img: Image.Image) -> Dict[str, Any]:
    """
    Route image through OCR pipeline.
    1. Check if blank → return empty
    2. Try standard Vision API (handwriting mode)
    3. If non-Latin detected → try multilingual with hints
    4. Return best result

    Args:
        img: PIL Image of an answer region.

    Returns:
        Dict with keys: text, confidence, engine, languages, is_blank.
    """
    result = {
        "text": "",
        "confidence": 0.0,
        "engine": "none",
        "languages": ["en"],
        "is_blank": False,
    }

    # Step 1: Blank check
    if is_blank(img):
        result["is_blank"] = True
        result["engine"] = "blank_detector"
        logger.info("Region detected as blank")
        return result

    # Step 2: Primary OCR — handwriting mode (works for both handwritten and printed)
    try:
        text, confidence = extract_handwritten_text(img)
        result["text"] = text
        result["confidence"] = confidence
        result["engine"] = "google_vision_handwriting"
    except Exception as e:
        logger.warning(f"Primary OCR failed: {e}")

    # Step 3: Check for non-Latin scripts
    if result["text"] and has_non_latin(result["text"]):
        logger.info("Non-Latin script detected, trying multilingual OCR")
        try:
            ml_text, ml_conf, ml_langs = extract_multilingual_text(img)
            if ml_conf > result["confidence"] or len(ml_text) > len(result["text"]):
                result["text"] = ml_text
                result["confidence"] = ml_conf
                result["engine"] = "google_vision_multilingual"
                result["languages"] = ml_langs if ml_langs else ["en"]
        except Exception as e:
            logger.warning(f"Multilingual OCR failed: {e}")

    # Step 4: If no text and not blank, try multilingual as fallback
    if not result["text"]:
        try:
            text, confidence, langs = extract_multilingual_text(img)
            if text:
                result["text"] = text
                result["confidence"] = confidence
                result["engine"] = "google_vision_multilingual_fallback"
                result["languages"] = langs if langs else ["en"]
        except Exception as e:
            logger.warning(f"Fallback OCR failed: {e}")

    logger.info(
        f"OCR result: engine={result['engine']}, "
        f"confidence={result['confidence']:.3f}, "
        f"text_length={len(result['text'])}"
    )
    return result
