"""
Course Processor: Process an entire course — all student PDFs.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from ingestion.pdf_converter import pdf_to_images
from preprocessing.scan_enhancer import enhance_scan
from preprocessing.region_segmenter import segment_regions
from preprocessing.question_mapper import map_regions_to_questions, merge_question_regions
from ocr.ocr_router import route_ocr
from evaluation.few_shot_builder import build_few_shot_examples, get_few_shot_messages
from processing.student_processor import process_student
from database.db_manager import DatabaseManager
from utils.logger import get_logger

logger = get_logger("course_processor")


class CourseProcessor:
    """Process an entire course assessment."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def process(
        self,
        course_code: str,
        assessment_title: str,
        answer_key_path: str,
        students_dir: str,
        marks_per_question: List[float] = None,
        samples_dir: str = None,
        grade_boundaries: Dict = None,
        progress_callback: Callable = None,
    ) -> Dict[str, Any]:
        """
        Process an entire course assessment.

        Args:
            course_code: Course code (e.g., "CS101").
            assessment_title: Assessment title (e.g., "Midterm").
            answer_key_path: Path to answer key PDF.
            students_dir: Directory containing student PDFs.
            marks_per_question: List of marks per question.
            samples_dir: Optional directory with teacher-marked samples.
            grade_boundaries: Optional grade boundaries.
            progress_callback: Optional callback(message).

        Returns:
            Summary dict with processing results.
        """
        def _progress(msg):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        summary = {
            "course_code": course_code,
            "assessment": assessment_title,
            "students_processed": 0,
            "students_total": 0,
            "errors": [],
        }

        try:
            # Step 1: Get or create course
            _progress(f"Setting up course: {course_code}")
            course = self.db.create_course(name=course_code, code=course_code)
            course_id = course.id if hasattr(course, 'id') else course['id'] if isinstance(course, dict) else course

            # Step 2: Process answer key
            _progress("Processing answer key...")
            answer_key_texts, answer_key_images = self._process_answer_key(answer_key_path)

            # Auto-detect question count if marks not specified
            if not marks_per_question:
                n_questions = max(len(answer_key_texts), 1)
                marks_per_question = [10.0] * n_questions
                _progress(f"Auto-detected {n_questions} questions, 10 marks each")

            total_marks = sum(marks_per_question)

            # Step 3: Create assessment
            assessment = self.db.create_assessment(
                course_id=course_id,
                title=assessment_title,
                total_marks=total_marks,
                answer_key_path=answer_key_path,
                marks_per_question=marks_per_question,
                grade_boundaries=grade_boundaries,
            )
            assessment_id = assessment.id if hasattr(assessment, 'id') else assessment['id'] if isinstance(assessment, dict) else assessment

            # Step 4: Build few-shot examples if samples provided
            few_shot_messages = None
            if samples_dir and Path(samples_dir).exists():
                _progress("Building few-shot examples from teacher samples...")
                sample_pdfs = list(Path(samples_dir).glob("*.pdf"))
                if sample_pdfs:
                    examples = build_few_shot_examples(
                        [str(p) for p in sample_pdfs]
                    )
                    few_shot_messages = get_few_shot_messages(examples)
                    _progress(f"Built {len(few_shot_messages)//2} few-shot examples")

            # Step 5: Discover student PDFs
            students_path = Path(students_dir)
            pdf_files = sorted(students_path.glob("*.pdf"))
            summary["students_total"] = len(pdf_files)
            _progress(f"Found {len(pdf_files)} student PDFs")

            # Step 6: Process each student
            for idx, pdf_file in enumerate(pdf_files):
                student_name = pdf_file.stem
                roll_number = self._extract_roll_number(student_name)

                _progress(
                    f"Processing student {idx + 1}/{len(pdf_files)}: {student_name}"
                )

                try:
                    # Create student record
                    student = self.db.create_student(
                        name=student_name,
                        roll_number=roll_number,
                        course_id=course_id,
                    )
                    student_id = student.id if hasattr(student, 'id') else student['id'] if isinstance(student, dict) else student

                    # Create result record
                    result = self.db.create_result(
                        student_id=student_id,
                        assessment_id=assessment_id,
                        pdf_path=str(pdf_file),
                    )
                    result_id = result.id if hasattr(result, 'id') else result['id'] if isinstance(result, dict) else result

                    # Process student
                    student_summary = process_student(
                        pdf_path=str(pdf_file),
                        result_id=result_id,
                        answer_key_texts=answer_key_texts,
                        marks_per_question=marks_per_question,
                        db=self.db,
                        few_shot_messages=few_shot_messages,
                        answer_key_images=answer_key_images,
                        progress_callback=lambda msg: _progress(f"  [{student_name}] {msg}"),
                        grade_boundaries=grade_boundaries,
                    )

                    if student_summary["status"] == "completed":
                        summary["students_processed"] += 1
                    else:
                        summary["errors"].append(f"{student_name}: processing error")

                except Exception as e:
                    logger.error(f"Failed to process {student_name}: {e}")
                    summary["errors"].append(f"{student_name}: {str(e)}")

            _progress(
                f"\nProcessing complete: {summary['students_processed']}/{summary['students_total']} students"
            )
            if summary["errors"]:
                _progress(f"Errors: {len(summary['errors'])}")

        except Exception as e:
            logger.error(f"Course processing failed: {e}")
            summary["errors"].append(f"Fatal error: {str(e)}")
            _progress(f"Error: {str(e)}")

        return summary

    def _process_answer_key(self, answer_key_path: str):
        """Process answer key PDF into per-question texts and images."""
        answer_key_texts = {}
        answer_key_images = {}

        try:
            images = pdf_to_images(answer_key_path)
            all_regions = []
            for page_img in images:
                enhanced = enhance_scan(page_img)
                regions = segment_regions(enhanced)
                all_regions.extend(regions)

            # OCR all regions
            ocr_texts = []
            for region in all_regions:
                ocr_result = route_ocr(region)
                ocr_texts.append(ocr_result["text"])

            # Map to questions
            question_map = map_regions_to_questions(all_regions, ocr_texts)
            question_images_merged = merge_question_regions(question_map)

            for q_num, img in question_images_merged.items():
                answer_key_images[q_num] = img
                # OCR the merged image for text
                ocr_result = route_ocr(img)
                answer_key_texts[q_num] = ocr_result["text"]

            logger.info(f"Answer key: {len(answer_key_texts)} questions extracted")

        except Exception as e:
            logger.error(f"Failed to process answer key: {e}")

        return answer_key_texts, answer_key_images

    def _extract_roll_number(self, filename: str) -> str:
        """Extract roll number from filename."""
        import re
        # Try patterns like: "123_StudentName", "ROLL123", "2024CS101"
        match = re.search(r"(\d{4,}[A-Z]*\d*)", filename)
        if match:
            return match.group(1)
        # Fallback: use filename
        return filename.replace(" ", "_")
