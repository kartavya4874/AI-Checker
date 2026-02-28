"""
Math Analyzer: GPT-4o Vision extracts LaTeX → SymPy symbolic comparison with answer key.
"""

import json
import re
from PIL import Image
from typing import Dict, Any, List
from openai import OpenAI
from sympy import simplify, sympify
from sympy.parsing.latex import parse_latex
from config import config
from utils.image_utils import image_to_base64
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("math_analyzer")


def analyze_math_answer(
    student_img: Image.Image,
    answer_key_text: str,
    marks_allocated: float,
) -> Dict[str, Any]:
    """
    Analyze a math answer using GPT-4o LaTeX extraction + SymPy comparison.

    Args:
        student_img: PIL Image of student's math answer.
        answer_key_text: Expected answer (LaTeX or text).
        marks_allocated: Total marks for this question.

    Returns:
        Dict with latex_extracted, comparison_results, step_analysis,
        suggested_marks, confidence.
    """
    result = {
        "latex_extracted": "",
        "comparison_results": [],
        "step_analysis": [],
        "suggested_marks": 0.0,
        "confidence": 0.0,
        "method": "math_analyzer",
    }

    try:
        # Step 1: Extract LaTeX from student image using GPT-4o Vision
        student_latex = _extract_latex_from_image(student_img)
        result["latex_extracted"] = student_latex

        if not student_latex.strip():
            result["step_analysis"].append("No mathematical content detected")
            return result

        # Step 2: Parse and compare with SymPy
        student_steps = _split_steps(student_latex)
        answer_steps = _split_steps(answer_key_text)

        step_results = []
        correct_steps = 0

        for i, s_step in enumerate(student_steps):
            step_result = {"step": i + 1, "student": s_step, "correct": False, "note": ""}

            # Compare with corresponding answer step
            if i < len(answer_steps):
                is_equiv = _sympy_compare(s_step, answer_steps[i])
                step_result["correct"] = is_equiv
                step_result["note"] = "Equivalent" if is_equiv else "Different from expected"
            else:
                # Extra steps — check if equivalent to final answer
                if answer_steps:
                    is_equiv = _sympy_compare(s_step, answer_steps[-1])
                    step_result["correct"] = is_equiv
                    step_result["note"] = "Matches final answer" if is_equiv else "Extra step"

            if step_result["correct"]:
                correct_steps += 1
            step_results.append(step_result)

        result["step_analysis"] = step_results

        # Also check if final student expression matches final answer
        if student_steps and answer_steps:
            final_match = _sympy_compare(student_steps[-1], answer_steps[-1])
            result["comparison_results"].append({
                "type": "final_answer",
                "match": final_match,
            })

        # Calculate suggested marks (partial credit)
        if step_results:
            ratio = correct_steps / max(len(answer_steps), len(student_steps), 1)
            result["suggested_marks"] = round(marks_allocated * ratio, 1)
            result["confidence"] = 0.8 if correct_steps > 0 else 0.5
        else:
            result["confidence"] = 0.3

    except Exception as e:
        logger.error(f"Math analysis failed: {e}")
        result["step_analysis"].append(f"Analysis error: {str(e)}")
        result["confidence"] = 0.2

    return result


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def _extract_latex_from_image(img: Image.Image) -> str:
    """Use GPT-4o Vision to extract LaTeX from handwritten math."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    b64 = image_to_base64(img)

    response = client.chat.completions.create(
        model=config.OPENAI_EVAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL mathematical expressions from this handwritten image as LaTeX. "
                            "Return ONLY the LaTeX, one expression per line. No explanation."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=1000,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


def _split_steps(latex_text: str) -> List[str]:
    """Split LaTeX text into individual expressions/steps."""
    lines = latex_text.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip("$ \\")
        if line:
            steps.append(line)
    return steps


def _sympy_compare(expr1: str, expr2: str) -> bool:
    """Compare two expressions using SymPy symbolic equivalence."""
    try:
        # Try parsing as LaTeX first
        try:
            sym1 = parse_latex(expr1)
        except Exception:
            sym1 = sympify(expr1)

        try:
            sym2 = parse_latex(expr2)
        except Exception:
            sym2 = sympify(expr2)

        diff = simplify(sym1 - sym2)
        return diff == 0 or diff.is_zero

    except Exception as e:
        logger.debug(f"SymPy comparison failed for '{expr1}' vs '{expr2}': {e}")
        # Fallback: string comparison
        return expr1.strip().replace(" ", "") == expr2.strip().replace(" ", "")
