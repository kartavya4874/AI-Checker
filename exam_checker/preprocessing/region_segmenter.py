"""
Segment answer sheet into question regions using contour/line detection.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
from utils.image_utils import pil_to_numpy, numpy_to_pil
from utils.logger import get_logger

logger = get_logger("region_segmenter")


def segment_regions(img: Image.Image, min_height_ratio: float = 0.03) -> List[Image.Image]:
    """
    Segment an answer sheet image into question regions.
    Uses horizontal line detection to find dividers between answers.

    Args:
        img: PIL Image of a full answer sheet page.
        min_height_ratio: Minimum region height as ratio of page height.

    Returns:
        List of PIL Image crops for each detected region.
    """
    try:
        arr = pil_to_numpy(img)
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape) == 3 else arr

        # Detect horizontal lines
        divider_ys = _detect_horizontal_dividers(gray, w, h)

        if not divider_ys:
            # Fallback: try projection profile segmentation
            divider_ys = _projection_profile_segment(gray, h)

        if not divider_ys:
            # No dividers found — return entire image as one region
            logger.info("No dividers detected, returning full page as single region")
            return [img]

        # Add top and bottom boundaries
        divider_ys = sorted(set([0] + divider_ys + [h]))

        # Extract regions
        min_h = int(h * min_height_ratio)
        regions = []
        for i in range(len(divider_ys) - 1):
            y1, y2 = divider_ys[i], divider_ys[i + 1]
            if (y2 - y1) >= min_h:
                crop = img.crop((0, y1, w, y2))
                regions.append(crop)

        logger.info(f"Segmented page into {len(regions)} regions")
        return regions

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return [img]


def _detect_horizontal_dividers(gray: np.ndarray, w: int, h: int) -> List[int]:
    """Detect horizontal lines that divide answer regions."""
    # Create horizontal kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Find contours of detected lines
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    divider_ys = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        # Line must span at least 40% of page width
        if cw >= w * 0.4:
            divider_ys.append(y + ch // 2)

    return sorted(divider_ys)


def _projection_profile_segment(gray: np.ndarray, h: int) -> List[int]:
    """Fallback segmentation using horizontal projection profile."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = np.sum(binary, axis=1)

    # Normalize
    max_val = max(projection) if max(projection) > 0 else 1
    projection = projection / max_val

    # Find valleys (low ink density rows)
    threshold = 0.05
    min_gap = int(h * 0.02)

    dividers = []
    in_gap = False
    gap_start = 0

    for i, val in enumerate(projection):
        if val < threshold:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gap_length = i - gap_start
                if gap_length >= min_gap:
                    dividers.append(gap_start + gap_length // 2)
                in_gap = False

    return dividers


def get_region_bboxes(img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Return bounding boxes of detected regions as (x1, y1, x2, y2).
    """
    arr = pil_to_numpy(img)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape) == 3 else arr

    divider_ys = _detect_horizontal_dividers(gray, w, h)
    if not divider_ys:
        divider_ys = _projection_profile_segment(gray, h)
    if not divider_ys:
        return [(0, 0, w, h)]

    divider_ys = sorted(set([0] + divider_ys + [h]))
    min_h = int(h * 0.03)
    bboxes = []
    for i in range(len(divider_ys) - 1):
        y1, y2 = divider_ys[i], divider_ys[i + 1]
        if (y2 - y1) >= min_h:
            bboxes.append((0, y1, w, y2))

    return bboxes
