"""
Diagram Analyzer: GPT-4o Vision evaluates diagram accuracy.
"""

import json
from PIL import Image
from typing import Dict, Any
from openai import OpenAI
from config import config
from utils.image_utils import image_to_base64
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("diagram_analyzer")


def analyze_diagram(
    student_img: Image.Image,
    description: str,
    marks_allocated: float,
    answer_key_img: Image.Image = None,
) -> Dict[str, Any]:
    """
    Evaluate a student diagram using GPT-4o Vision.

    Args:
        student_img: PIL Image of student's diagram.
        description: Expected diagram description from answer key.
        marks_allocated: Total marks for this question.
        answer_key_img: Optional PIL Image of reference diagram.

    Returns:
        Dict with elements_present, elements_missing, labels_correct,
        overall_accuracy, feedback, suggested_marks.
    """
    result = {
        "elements_present": [],
        "elements_missing": [],
        "labels_correct": [],
        "overall_accuracy": 0.0,
        "feedback": "",
        "suggested_marks": 0.0,
        "confidence": 0.0,
        "method": "diagram_analyzer",
    }

    try:
        eval_result = _evaluate_diagram_gpt4o(
            student_img, description, answer_key_img
        )

        if eval_result:
            result.update(eval_result)
            result["suggested_marks"] = round(
                marks_allocated * result.get("overall_accuracy", 0), 1
            )
            result["confidence"] = 0.7

    except Exception as e:
        logger.error(f"Diagram analysis failed: {e}")
        result["feedback"] = f"Analysis error: {str(e)}"
        result["confidence"] = 0.2

    return result


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def _evaluate_diagram_gpt4o(
    student_img: Image.Image,
    description: str,
    answer_key_img: Image.Image = None,
) -> Dict[str, Any]:
    """Use GPT-4o Vision to evaluate diagram."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    content = [
        {
            "type": "text",
            "text": (
                f"Evaluate this student diagram against the description: {description}\n"
                "Return JSON with:\n"
                '- "elements_present": list of correctly drawn elements\n'
                '- "elements_missing": list of missing elements\n'
                '- "labels_correct": list of correctly labeled parts\n'
                '- "overall_accuracy": float 0-1\n'
                '- "feedback": string with detailed evaluation\n'
                "Return ONLY valid JSON."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_to_base64(student_img)}"
            },
        },
    ]

    # Include answer key image if available
    if answer_key_img:
        content.append({
            "type": "text",
            "text": "Reference diagram (answer key):",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_to_base64(answer_key_img)}"
            },
        })

    response = client.chat.completions.create(
        model=config.OPENAI_EVAL_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=800,
        temperature=0.0,
    )

    response_text = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        return json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse diagram evaluation JSON: {response_text[:200]}")
        return {
            "elements_present": [],
            "elements_missing": [],
            "labels_correct": [],
            "overall_accuracy": 0.5,
            "feedback": response_text,
        }
