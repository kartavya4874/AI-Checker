"""
Evaluator Orchestrator: Full pipeline per question.
OCR → classify → analyzer → GPT-4o final evaluation.
"""

from PIL import Image
from typing import Dict, Any, List
from ocr.ocr_router import route_ocr
from evaluation.content_classifier import classify_content
from evaluation.gpt4o_evaluator import evaluate_answer
from content_analyzers.math_analyzer import analyze_math_answer
from content_analyzers.chemistry_analyzer import analyze_chemistry_answer
from content_analyzers.diagram_analyzer import analyze_diagram
from content_analyzers.code_analyzer import analyze_code
from content_analyzers.text_analyzer import analyze_text_answer
from utils.logger import get_logger

logger = get_logger("evaluator_orchestrator")


def evaluate_question(
    question_number: int,
    student_img: Image.Image,
    answer_key_text: str,
    marks_allocated: float,
    few_shot_messages: List[Dict] = None,
    answer_key_img: Image.Image = None,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a single question.

    Pipeline: OCR → classify → analyzer → GPT-4o evaluate.

    Args:
        question_number: Question number.
        student_img: PIL Image of student's answer.
        answer_key_text: Expected answer text.
        marks_allocated: Total marks available.
        few_shot_messages: Optional few-shot examples.
        answer_key_img: Optional answer key image.

    Returns:
        Complete evaluation result dict.
    """
    result = {
        "question_number": question_number,
        "marks_allocated": marks_allocated,
        "marks_obtained": 0.0,
        "content_type": "text",
        "status": "unattempted",
        "ocr_text": "",
        "ocr_engine": "",
        "feedback": "",
        "error_analysis": "",
        "partial_credit_breakdown": [],
        "pre_analysis": {},
    }

    try:
        # Step 1: OCR
        logger.info(f"Q{question_number}: Running OCR...")
        ocr_result = route_ocr(student_img)
        result["ocr_text"] = ocr_result["text"]
        result["ocr_engine"] = ocr_result["engine"]

        # Check if blank
        if ocr_result["is_blank"] or not ocr_result["text"].strip():
            result["status"] = "unattempted"
            result["feedback"] = "Question not attempted."
            result["marks_obtained"] = 0.0
            logger.info(f"Q{question_number}: Not attempted (blank)")
            return result

        result["status"] = "attempted"

        # Step 2: Classify content type
        logger.info(f"Q{question_number}: Classifying content...")
        content_type = classify_content(ocr_result["text"], student_img)
        result["content_type"] = content_type

        # Step 3: Run appropriate analyzer
        logger.info(f"Q{question_number}: Running {content_type} analyzer...")
        pre_analysis = _run_analyzer(
            content_type, student_img, ocr_result["text"],
            answer_key_text, marks_allocated, answer_key_img
        )
        result["pre_analysis"] = pre_analysis

        # Step 4: GPT-4o final evaluation
        logger.info(f"Q{question_number}: Running GPT-4o final evaluation...")
        final_eval = evaluate_answer(
            question_number=question_number,
            student_img=student_img,
            ocr_text=ocr_result["text"],
            content_type=content_type,
            answer_key_text=answer_key_text,
            marks_allocated=marks_allocated,
            pre_analysis=pre_analysis,
            few_shot_messages=few_shot_messages,
        )

        result["marks_obtained"] = final_eval["marks_obtained"]
        result["feedback"] = final_eval["feedback"]
        result["error_analysis"] = final_eval["error_analysis"]
        result["partial_credit_breakdown"] = final_eval["partial_credit_breakdown"]

        # Determine status
        if result["marks_obtained"] >= marks_allocated:
            result["status"] = "attempted"
        elif result["marks_obtained"] > 0:
            result["status"] = "partial"
        else:
            result["status"] = "attempted"

        logger.info(
            f"Q{question_number}: {result['marks_obtained']}/{marks_allocated} "
            f"({content_type})"
        )

    except Exception as e:
        logger.error(f"Q{question_number} evaluation failed: {e}")
        result["feedback"] = f"Evaluation error: {str(e)}"
        result["status"] = "error"

    return result


def _run_analyzer(
    content_type: str,
    student_img: Image.Image,
    ocr_text: str,
    answer_key_text: str,
    marks_allocated: float,
    answer_key_img: Image.Image = None,
) -> Dict[str, Any]:
    """Run the appropriate content analyzer."""
    try:
        if content_type == "math":
            return analyze_math_answer(student_img, answer_key_text, marks_allocated)

        elif content_type == "chemistry_structure":
            return analyze_chemistry_answer(
                student_img, answer_key_text, marks_allocated, "structure"
            )

        elif content_type == "chemistry_equation":
            return analyze_chemistry_answer(
                student_img, answer_key_text, marks_allocated, "equation"
            )

        elif content_type == "diagram":
            return analyze_diagram(
                student_img, answer_key_text, marks_allocated, answer_key_img
            )

        elif content_type == "code":
            return analyze_code(ocr_text, answer_key_text, marks_allocated)

        elif content_type == "mixed":
            # For mixed content, run text analyzer as primary
            return analyze_text_answer(ocr_text, answer_key_text, marks_allocated)

        else:  # "text" or default
            return analyze_text_answer(ocr_text, answer_key_text, marks_allocated)

    except Exception as e:
        logger.warning(f"Analyzer failed for {content_type}: {e}")
        return {"error": str(e), "method": content_type}
