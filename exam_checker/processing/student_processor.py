"""
Student Processor: Process a single student's answer sheet.
PDF → images → enhance → segment → OCR → evaluate → grade → persist.
"""

from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from PIL import Image
from ingestion.pdf_converter import pdf_to_images
from preprocessing.scan_enhancer import enhance_scan
from preprocessing.region_segmenter import segment_regions
from preprocessing.question_mapper import map_regions_to_questions, merge_question_regions
from ocr.ocr_router import route_ocr
from evaluation.evaluator_orchestrator import evaluate_question
from evaluation.grading_engine import calculate_grade, calculate_percentage
from database.db_manager import DatabaseManager
from utils.logger import get_logger

logger = get_logger("student_processor")


def process_student(
    pdf_path: str,
    result_id: int,
    answer_key_texts: Dict[int, str],
    marks_per_question: List[float],
    db: DatabaseManager,
    few_shot_messages: List[Dict] = None,
    answer_key_images: Dict[int, Image.Image] = None,
    progress_callback: Callable = None,
    grade_boundaries: Dict = None,
) -> Dict[str, Any]:
    """
    Process a single student's answer sheet through the full pipeline.

    Args:
        pdf_path: Path to student's answer sheet PDF.
        result_id: Database Result ID to update.
        answer_key_texts: Dict mapping question_number → answer key text.
        marks_per_question: List of marks allocated per question.
        db: DatabaseManager instance.
        few_shot_messages: Optional few-shot examples.
        answer_key_images: Optional answer key images per question.
        progress_callback: Optional callback(message) for progress updates.
        grade_boundaries: Optional grade boundaries dict.

    Returns:
        Summary dict with total_marks, percentage, grade.
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    summary = {
        "total_marks_obtained": 0.0,
        "total_marks_allocated": 0.0,
        "percentage": 0.0,
        "grade": "",
        "question_count": 0,
        "status": "error",
    }

    try:
        # Step 1: Convert PDF to images
        _progress("Converting PDF to images...")
        db.update_result(result_id, processing_status="processing")
        images = pdf_to_images(pdf_path)

        if not images:
            _progress("Error: No pages found in PDF")
            db.update_result(result_id, processing_status="error")
            return summary

        # Step 2: Enhance all pages
        _progress("Enhancing scans...")
        enhanced_images = [enhance_scan(img) for img in images]

        # Step 3: Segment all pages into regions
        _progress("Segmenting answer regions...")
        all_regions = []
        for page_img in enhanced_images:
            page_regions = segment_regions(page_img)
            all_regions.extend(page_regions)

        # Step 4: OCR all regions for question mapping
        _progress("Running OCR for question mapping...")
        ocr_texts = []
        for region in all_regions:
            ocr_result = route_ocr(region)
            ocr_texts.append(ocr_result["text"])

        # Step 5: Map regions to question numbers
        expected_count = len(marks_per_question)
        question_map = map_regions_to_questions(all_regions, ocr_texts, expected_count)
        question_images = merge_question_regions(question_map)

        # Step 6: Evaluate each question
        total_obtained = 0.0
        total_allocated = sum(marks_per_question)
        question_count = 0

        for q_num in sorted(question_images.keys()):
            q_idx = q_num - 1
            if q_idx < len(marks_per_question):
                marks = marks_per_question[q_idx]
            else:
                marks = marks_per_question[-1] if marks_per_question else 10.0

            answer_text = answer_key_texts.get(q_num, "")
            answer_img = answer_key_images.get(q_num) if answer_key_images else None

            _progress(f"Evaluating Question {q_num}...")

            q_result = evaluate_question(
                question_number=q_num,
                student_img=question_images[q_num],
                answer_key_text=answer_text,
                marks_allocated=marks,
                few_shot_messages=few_shot_messages,
                answer_key_img=answer_img,
            )

            # Save question result to DB
            db.save_question_result(result_id, q_result)
            total_obtained += q_result["marks_obtained"]
            question_count += 1

        # Handle unattempted questions
        for q_idx, marks in enumerate(marks_per_question):
            q_num = q_idx + 1
            if q_num not in question_images:
                db.save_question_result(result_id, {
                    "question_number": q_num,
                    "marks_allocated": marks,
                    "marks_obtained": 0,
                    "content_type": "text",
                    "status": "unattempted",
                    "feedback": "Question not attempted.",
                })

        # Step 7: Calculate grade
        percentage = calculate_percentage(total_obtained, total_allocated)
        grade = calculate_grade(total_obtained, total_allocated, grade_boundaries)

        # Step 8: Update result in DB
        db.update_result(
            result_id,
            total_marks_obtained=total_obtained,
            total_marks_allocated=total_allocated,
            percentage=percentage,
            grade=grade,
            processing_status="completed",
        )

        summary = {
            "total_marks_obtained": total_obtained,
            "total_marks_allocated": total_allocated,
            "percentage": percentage,
            "grade": grade,
            "question_count": question_count,
            "status": "completed",
        }

        _progress(
            f"Complete: {total_obtained}/{total_allocated} "
            f"({percentage}%) — Grade: {grade}"
        )

    except Exception as e:
        logger.error(f"Student processing failed: {e}")
        db.update_result(result_id, processing_status="error")
        summary["status"] = "error"
        _progress(f"Error: {str(e)}")

    return summary
