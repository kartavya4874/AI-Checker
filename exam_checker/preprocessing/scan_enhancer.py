"""
Scan enhancement: deskew, denoise, contrast using OpenCV.
"""

import cv2
import numpy as np
from PIL import Image
from utils.image_utils import pil_to_numpy, numpy_to_pil
from utils.logger import get_logger

logger = get_logger("scan_enhancer")


def enhance_scan(img: Image.Image) -> Image.Image:
    """
    Full enhancement pipeline: grayscale → deskew → denoise → contrast.

    Args:
        img: PIL Image to enhance.

    Returns:
        Enhanced PIL Image (RGB).
    """
    try:
        arr = pil_to_numpy(img)

        # Convert to grayscale for processing
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr.copy()

        # 1. Deskew
        gray = _deskew(gray)

        # 2. Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # 3. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 4. Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)

        # Convert back to RGB
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return numpy_to_pil(result)

    except Exception as e:
        logger.warning(f"Enhancement failed, returning original: {e}")
        return img


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Deskew image using Hough line detection."""
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:  # Only near-horizontal lines
                angles.append(angle)

        if not angles:
            return gray

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return gray

        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(gray, rotation_matrix, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        logger.debug(f"Deskewed by {median_angle:.2f}°")
        return rotated

    except Exception as e:
        logger.debug(f"Deskew failed: {e}")
        return gray
