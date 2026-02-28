"""
GPT-4o Final Evaluator: Combines all pre-analysis signals for final evaluation.
Returns JSON with marks, feedback, error analysis, partial credit breakdown.
"""

import json
from PIL import Image
from typing import Dict, Any, List
from openai import OpenAI
from config import config
from utils.image_utils import image_to_base64
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("gpt4o_evaluator")

SYSTEM_PROMPT = (
    "You are an expert university examiner with 20 years of experience. "
    "You receive pre-analyzed signals from specialized AI models "
    "(Math: SymPy equivalence, Chemistry: RDKit Tanimoto, Diagrams: structural analysis, "
    "Code: static analysis + execution, Text: embedding similarity). "
    "Make the FINAL evaluation using all signals. "
    "STRICT RULES: "
    "Match by question number not position. "
    "Not Attempted = 0 marks. "
    "Alternative correct approaches get FULL marks. "
    "Partial credit must be granular and justified. "
    "NEVER penalize messy handwriting. "
    "Always return valid JSON."
)


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def evaluate_answer(
    question_number: int,
    student_img: Image.Image,
    ocr_text: str,
    content_type: str,
    answer_key_text: str,
    marks_allocated: float,
    pre_analysis: Dict[str, Any],
    few_shot_messages: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Final GPT-4o evaluation combining all pre-analysis signals.

    Args:
        question_number: Question number.
        student_img: PIL Image of student's answer.
        ocr_text: OCR-extracted text.
        content_type: Classified content type.
        answer_key_text: Expected answer.
        marks_allocated: Total marks for this question.
        pre_analysis: Results from content analyzers.
        few_shot_messages: Optional few-shot example messages.

    Returns:
        Dict with marks_obtained, feedback, error_analysis,
        partial_credit_breakdown.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples
    if few_shot_messages:
        messages.extend(few_shot_messages[:4])

    # Build evaluation request
    eval_request = _build_evaluation_prompt(
        question_number, ocr_text, content_type,
        answer_key_text, marks_allocated, pre_analysis
    )

    # Include student image
    content = [
        {"type": "text", "text": eval_request},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_to_base64(student_img)}"
            },
        },
    ]

    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=config.OPENAI_EVAL_MODEL,
        messages=messages,
        max_tokens=800,
        temperature=0.0,
    )

    response_text = response.choices[0].message.content.strip()

    # Parse JSON response
    result = _parse_evaluation_response(response_text, marks_allocated)
    logger.info(
        f"Q{question_number}: {result['marks_obtained']}/{marks_allocated} marks"
    )
    return result


def _build_evaluation_prompt(
    question_number: int,
    ocr_text: str,
    content_type: str,
    answer_key_text: str,
    marks_allocated: float,
    pre_analysis: Dict[str, Any],
) -> str:
    """Build the evaluation prompt with all context."""
    prompt = f"""EVALUATE Question {question_number} ({content_type})
Marks Available: {marks_allocated}

--- Student Answer (OCR) ---
{ocr_text[:2000] if ocr_text else '[No text detected]'}

--- Answer Key ---
{answer_key_text[:2000]}

--- Pre-Analysis Signals ---
"""
    # Add relevant pre-analysis data
    for key, value in pre_analysis.items():
        if isinstance(value, dict):
            prompt += f"\n{key}:\n"
            for k, v in value.items():
                prompt += f"  {k}: {v}\n"
        else:
            prompt += f"{key}: {value}\n"

    prompt += """
Return JSON with:
- "marks_obtained": number (0 to marks_allocated)
- "feedback": string (detailed constructive feedback)
- "error_analysis": string (description of errors/misconceptions)
- "partial_credit_breakdown": list of {"component": string, "marks": number, "reason": string}
"""
    return prompt


def _parse_evaluation_response(
    response_text: str, marks_allocated: float
) -> Dict[str, Any]:
    """Parse GPT-4o evaluation response into structured dict."""
    default = {
        "marks_obtained": 0,
        "feedback": "",
        "error_analysis": "",
        "partial_credit_breakdown": [],
    }

    try:
        # Clean JSON from markdown blocks
        text = response_text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        parsed = json.loads(text)

        # Validate marks
        marks = float(parsed.get("marks_obtained", 0))
        marks = max(0, min(marks, marks_allocated))

        return {
            "marks_obtained": marks,
            "feedback": str(parsed.get("feedback", "")),
            "error_analysis": str(parsed.get("error_analysis", "")),
            "partial_credit_breakdown": parsed.get("partial_credit_breakdown", []),
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse evaluation JSON: {e}")
        default["feedback"] = response_text[:500]
        return default
