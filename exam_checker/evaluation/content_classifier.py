"""
Classify answer region content type using regex + OpenCV heuristics.
Types: text, math, chemistry_structure, chemistry_equation, diagram, code, mixed.
"""

import re
import cv2
import numpy as np
from PIL import Image
from utils.image_utils import pil_to_numpy
from utils.logger import get_logger

logger = get_logger("content_classifier")

# Regex patterns for content type detection
MATH_PATTERNS = [
    r"[=+\-*/^√∑∫∂∞≠≤≥±]",
    r"\\(?:frac|sqrt|int|sum|lim|sin|cos|tan|log|ln)",
    r"\d+\s*[+\-*/=]\s*\d+",
    r"[xy]\s*[=<>]\s*",
    r"d[xy]/d[xy]",
]

CHEMISTRY_PATTERNS = [
    r"\b(?:H2O|CO2|NaCl|HCl|H2SO4|NaOH|CH4)\b",
    r"\b\d*[A-Z][a-z]?\d*\s*\+\s*\d*[A-Z]",
    r"→|⟶|->|yields",
    r"\b(?:mol|molar|pH|pKa|ΔH|ΔG)\b",
]

CODE_PATTERNS = [
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfor\s+\w+\s+in\s+",
    r"\bwhile\s*\(",
    r"\bif\s*\(.*\)\s*\{",
    r"console\.log|System\.out|printf|cout",
    r"def\s|return\s|print\(",
]


def classify_content(
    ocr_text: str,
    img: Image.Image = None,
) -> str:
    """
    Classify content type of an answer region.

    Args:
        ocr_text: OCR-extracted text from the region.
        img: PIL Image of the region (for visual heuristics).

    Returns:
        One of: "text", "math", "chemistry_structure", "chemistry_equation",
        "diagram", "code", "mixed".
    """
    scores = {
        "text": 0.0,
        "math": 0.0,
        "chemistry_structure": 0.0,
        "chemistry_equation": 0.0,
        "diagram": 0.0,
        "code": 0.0,
    }

    if not ocr_text.strip() and img is not None:
        # No text but has content → likely diagram
        if _has_diagram_features(img):
            return "diagram"
        return "text"

    # Text analysis
    text = ocr_text.strip()
    word_count = len(text.split())
    scores["text"] = word_count * 0.1

    # Math detection
    for pattern in MATH_PATTERNS:
        matches = len(re.findall(pattern, text))
        scores["math"] += matches * 2.0

    # Chemistry detection
    for pattern in CHEMISTRY_PATTERNS:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        scores["chemistry_equation"] += matches * 3.0

    # Code detection
    for pattern in CODE_PATTERNS:
        matches = len(re.findall(pattern, text))
        scores["code"] += matches * 3.0

    # Image-based heuristics
    if img is not None:
        if _has_diagram_features(img):
            scores["diagram"] += 5.0

        # Chemical structure heuristic: many lines, few text
        if _has_structure_features(img, text):
            scores["chemistry_structure"] += 5.0

    # Determine winner
    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]

    if max_score < 1.0:
        return "text"  # Default

    # Check for mixed content
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        if sorted_scores[1][1] > 0.6 * sorted_scores[0][1]:
            logger.debug(f"Mixed content detected: {sorted_scores[0][0]} + {sorted_scores[1][0]}")
            return "mixed"

    logger.debug(f"Content classified as: {max_type} (scores: {scores})")
    return max_type


def _has_diagram_features(img: Image.Image) -> bool:
    """Detect diagram-like features using OpenCV contour analysis."""
    try:
        arr = pil_to_numpy(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape) == 3 else arr
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Diagrams tend to have large closed contours
        h, w = gray.shape
        large_contours = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.01):  # > 1% of image area
                large_contours += 1

        # Check for geometric shapes
        geometric_shapes = 0
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if 3 <= len(approx) <= 8:
                geometric_shapes += 1

        return large_contours >= 2 or geometric_shapes >= 3

    except Exception:
        return False


def _has_structure_features(img: Image.Image, text: str) -> bool:
    """Detect chemical structure features."""
    # Chemical structures have lines/bonds but very little text
    text_density = len(text.split()) / max(1, img.height * img.width / 10000)
    has_chem_keywords = bool(re.search(r"[A-Z][a-z]?(?:\d|$)", text))
    return text_density < 1.0 and has_chem_keywords
