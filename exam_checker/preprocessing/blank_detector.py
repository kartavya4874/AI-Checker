"""
Detect blank/unanswered regions using ink ratio analysis.
"""

import cv2
import numpy as np
from PIL import Image
from utils.image_utils import pil_to_numpy
from utils.logger import get_logger

logger = get_logger("blank_detector")


def is_blank(img: Image.Image, threshold: float = 0.01) -> bool:
    """
    Detect if a region is blank (unanswered) based on ink density.

    Args:
        img: PIL Image of the region.
        threshold: Minimum dark pixel ratio to consider as non-blank.

    Returns:
        True if the region is blank.
    """
    try:
        arr = pil_to_numpy(img)

        # Convert to grayscale
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr

        # Apply Otsu threshold to separate ink from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate ink ratio
        total_pixels = binary.shape[0] * binary.shape[1]
        dark_pixels = np.count_nonzero(binary)
        ink_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

        is_empty = ink_ratio < threshold
        logger.debug(f"Ink ratio: {ink_ratio:.4f}, threshold: {threshold}, blank: {is_empty}")
        return is_empty

    except Exception as e:
        logger.warning(f"Blank detection failed: {e}")
        return False


def get_ink_ratio(img: Image.Image) -> float:
    """Return the ink density ratio of the image (0.0 to 1.0)."""
    try:
        arr = pil_to_numpy(img)
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        total = binary.shape[0] * binary.shape[1]
        dark = np.count_nonzero(binary)
        return dark / total if total > 0 else 0.0

    except Exception:
        return 0.0
