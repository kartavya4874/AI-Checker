"""
Map detected regions to question numbers using regex on OCR text.
Supports multi-page answer sheets.
"""

import re
from typing import Dict, List, Optional, Tuple
from PIL import Image
from utils.logger import get_logger

logger = get_logger("question_mapper")

# Patterns to detect question numbers — robust against OCR noise
QUESTION_PATTERNS = [
    # Q1, Q.1, Que 1, Question 1, Ques-1 etc. (tolerant of leading noise)
    r"(?:^|\n)[^\n]{0,15}?\b(?:Q|Que|Question|Ques)[.\s:)#-]*(\d+)",
    # Ans 1, Answer 1, A.1 etc.
    r"(?:^|\n)[^\n]{0,15}?\b(?:Ans|Answer)[.\s:)#-]*(\d+)",
    # Standalone number with delimiter: "1.", "1)", "1 -" (must be at or near line start)
    r"(?:^|\n)\s{0,10}(\d+)\s*[.)]\s",
    # Parenthesized: (1), (2) etc.
    r"\((\d+)\)",
    # Number followed by dash: "1 -", "2 –"
    r"(?:^|\n)\s{0,10}(\d+)\s*[-–—]\s",
    # "Q.1" style with a dot between Q and number
    r"\bQ\.?\s*(\d+)\b",
]


def map_regions_to_questions(
    regions: List[Image.Image],
    ocr_texts: List[str],
    expected_count: int = None,
) -> Dict[int, List[Tuple[int, Image.Image]]]:
    """
    Map segmented regions to question numbers based on OCR text.

    Args:
        regions: List of region images from segmentation.
        ocr_texts: Corresponding OCR text for each region.
        expected_count: Expected number of questions (optional).

    Returns:
        Dict mapping question_number → [(region_index, region_image), ...]
        Supports multi-region per question (for multi-page answers).
    """
    question_map: Dict[int, List[Tuple[int, Image.Image]]] = {}
    current_question = None

    for idx, (region, text) in enumerate(zip(regions, ocr_texts)):
        detected_q = _extract_question_number(text)

        if detected_q is not None:
            current_question = detected_q
        elif current_question is None:
            # First region with no detected question number
            current_question = idx + 1

        if current_question not in question_map:
            question_map[current_question] = []
        question_map[current_question].append((idx, region))

    # If expected_count given and we have unassigned regions, assign sequentially
    if expected_count and len(question_map) < expected_count:
        assigned = set(question_map.keys())
        unassigned_regions = [
            (i, r) for i, r in enumerate(regions)
            if not any((i, r) in v for v in question_map.values())
        ]
        next_q = 1
        for i, r in unassigned_regions:
            while next_q in assigned:
                next_q += 1
            if next_q <= expected_count:
                question_map[next_q] = [(i, r)]
                assigned.add(next_q)

    logger.info(f"Mapped {len(regions)} regions to {len(question_map)} questions: {sorted(question_map.keys())}")
    return question_map


def _extract_question_number(text: str) -> Optional[int]:
    """Extract question number from OCR text."""
    if not text or not text.strip():
        return None

    # Try each pattern
    for pattern in QUESTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                q_num = int(match.group(1))
                if 1 <= q_num <= 100:  # Sanity check
                    return q_num
            except (ValueError, IndexError):
                continue

    return None


def merge_question_regions(
    question_map: Dict[int, List[Tuple[int, Image.Image]]]
) -> Dict[int, Image.Image]:
    """
    Merge multi-page regions for the same question into a single image.

    Args:
        question_map: Output from map_regions_to_questions.

    Returns:
        Dict mapping question_number → single merged PIL Image.
    """
    merged = {}
    for q_num, region_list in question_map.items():
        if len(region_list) == 1:
            merged[q_num] = region_list[0][1]
        else:
            # Vertically stack all regions for this question
            images = [r[1] for r in region_list]
            widths = [img.width for img in images]
            max_width = max(widths)
            total_height = sum(img.height for img in images)

            combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for img in images:
                combined.paste(img, (0, y_offset))
                y_offset += img.height

            merged[q_num] = combined
            logger.debug(f"Q{q_num}: merged {len(images)} regions → {max_width}x{total_height}")

    return merged
