"""
Few-Shot Builder: Parse teacher-marked sample papers for few-shot prompting.
"""

import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
from ingestion.pdf_converter import pdf_to_images
from preprocessing.scan_enhancer import enhance_scan
from ocr.ocr_router import route_ocr
from utils.image_utils import image_to_base64
from utils.logger import get_logger

logger = get_logger("few_shot_builder")


def build_few_shot_examples(
    sample_pdf_paths: List[str],
    answer_key_text: str = "",
) -> List[Dict]:
    """
    Parse teacher-marked sample papers to build few-shot examples.

    Args:
        sample_pdf_paths: Paths to teacher-marked sample PDFs.
        answer_key_text: The answer key text for context.

    Returns:
        List of few-shot message dicts for GPT-4o.
    """
    examples = []

    for pdf_path in sample_pdf_paths:
        try:
            path = Path(pdf_path)
            if not path.exists():
                logger.warning(f"Sample not found: {pdf_path}")
                continue

            logger.info(f"Processing sample: {path.name}")
            images = pdf_to_images(str(path))

            for i, img in enumerate(images):
                enhanced = enhance_scan(img)
                ocr_result = route_ocr(enhanced)
                text = ocr_result["text"]

                if not text.strip():
                    continue

                # Extract marks/feedback from teacher annotations
                marks_info = _extract_teacher_marks(text)

                if marks_info:
                    example = _create_few_shot_message(
                        text, marks_info, image_to_base64(enhanced)
                    )
                    examples.append(example)

        except Exception as e:
            logger.error(f"Failed to process sample {pdf_path}: {e}")

    logger.info(f"Built {len(examples)} few-shot examples")
    return examples


def _extract_teacher_marks(text: str) -> Dict:
    """Extract teacher marks and feedback annotations from OCR text."""
    import re

    marks_info = {}

    # Look for marks patterns like "5/10", "8 marks", "Score: 7"
    marks_patterns = [
        r"(\d+(?:\.\d+)?)\s*/\s*(\d+)",
        r"(?:marks?|score|grade):\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?)",
    ]

    for pattern in marks_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                marks_info["marks_given"] = float(groups[0])
                marks_info["marks_total"] = float(groups[1])
            else:
                marks_info["marks_given"] = float(groups[0])
            break

    # Look for feedback
    feedback_patterns = [
        r"(?:feedback|comment|note):\s*(.+?)(?:\n|$)",
        r"(?:good|excellent|poor|needs improvement|correct|incorrect|well done).*",
    ]

    for pattern in feedback_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            marks_info["feedback"] = match.group(0).strip()
            break

    return marks_info if marks_info else None


def _create_few_shot_message(
    student_text: str,
    marks_info: Dict,
    image_b64: str = None,
) -> Dict:
    """Create a few-shot user/assistant message pair."""
    user_content = f"Student answer:\n{student_text[:1000]}"

    assistant_response = {
        "marks_obtained": marks_info.get("marks_given", 0),
        "feedback": marks_info.get("feedback", "Teacher evaluated"),
        "partial_credit_breakdown": [],
    }

    return {
        "user": {"role": "user", "content": user_content},
        "assistant": {
            "role": "assistant",
            "content": json.dumps(assistant_response),
        },
    }


def get_few_shot_messages(examples: List[Dict]) -> List[Dict]:
    """Convert few-shot examples to flat message list."""
    messages = []
    for ex in examples[:3]:  # Limit to 3 examples
        messages.append(ex["user"])
        messages.append(ex["assistant"])
    return messages


def build_few_shot_from_past_results(
    db: Any, assessment_id: int, question_number: int
) -> List[Dict]:
    """
    Query the database for recently graded answers to use as dynamic few-shot examples.
    """
    past_results = db.get_successful_question_results(assessment_id, question_number, limit=2)
    examples = []
    
    for res in past_results:
        if not res.get("ocr_text"):
            continue
            
        user_content = f"Student answer:\n{res['ocr_text'][:1000]}"
        assistant_response = {
            "marks_obtained": res["marks_obtained"],
            "feedback": res["feedback"] or "Good response.",
            "partial_credit_breakdown": [],
        }
        
        examples.append({
            "user": {"role": "user", "content": user_content},
            "assistant": {
                "role": "assistant",
                "content": json.dumps(assistant_response),
            },
        })
        
    return examples
