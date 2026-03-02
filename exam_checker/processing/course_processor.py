"""
Course Processor: Process an entire course — all student PDFs.
Supports multithreaded student processing.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from ingestion.pdf_converter import pdf_to_images
from preprocessing.scan_enhancer import enhance_scan
from preprocessing.region_segmenter import segment_regions
from preprocessing.question_mapper import map_regions_to_questions, merge_question_regions
from preprocessing.answer_key_parser import parse_marks_from_answer_key
from ocr.ocr_router import route_ocr
from evaluation.few_shot_builder import build_few_shot_examples, get_few_shot_messages
from processing.student_processor import process_student
from database.db_manager import DatabaseManager
from utils.folder_scanner import scan_exam_folder, describe_scan, parse_exam_metadata
from config import config
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
        question_paper_path: str = None,
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
            question_paper_path: Optional path to the question paper PDF.

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
            answer_key_texts, answer_key_images, auto_marks = self._process_answer_key(answer_key_path)

            # Auto-detect marks if not manually provided
            if not marks_per_question:
                qp_marks = []
                if question_paper_path and Path(question_paper_path).exists():
                    _progress("Extracting marks from question paper...")
                    qp_marks = self._extract_marks_from_pdf(question_paper_path)
                
                if qp_marks:
                    marks_per_question = qp_marks
                    _progress(f"Auto-detected {len(marks_per_question)} questions with marks from question paper: {marks_per_question}")
                else:
                    marks_per_question = auto_marks
                    _progress(f"Auto-detected {len(marks_per_question)} questions with marks from answer key: {marks_per_question}")
            else:
                _progress(f"Using provided marks: {marks_per_question}")

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

            # Step 6: Process each student using multithreading
            max_workers = getattr(config, 'MAX_WORKERS', 4)
            _progress(f"Processing {len(pdf_files)} students with {max_workers} threads...")

            # Thread-safe counter
            lock = threading.Lock()

            def _process_one_student(idx_pdf):
                idx, pdf_file = idx_pdf
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

                    student_summary = process_student(
                        pdf_path=str(pdf_file),
                        result_id=result_id,
                        answer_key_texts=answer_key_texts,
                        marks_per_question=marks_per_question,
                        db=self.db,
                        few_shot_messages=few_shot_messages,
                        answer_key_images=answer_key_images,
                        progress_callback=lambda msg, sn=student_name: _progress(f"  [{sn}] {msg}"),
                        grade_boundaries=grade_boundaries,
                        assessment_id=assessment_id,
                    )

                    if student_summary["status"] == "completed":
                        with lock:
                            summary["students_processed"] += 1
                    else:
                        with lock:
                            summary["errors"].append(f"{student_name}: processing error")

                except Exception as e:
                    logger.error(f"Failed to process {student_name}: {e}")
                    with lock:
                        summary["errors"].append(f"{student_name}: {str(e)}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_one_student, (idx, pdf_file))
                    for idx, pdf_file in enumerate(pdf_files)
                ]
                for future in as_completed(futures):
                    # Propagate any uncaught exceptions
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Thread error: {e}")

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

    def process_from_folder(
        self,
        root_folder: str,
        course_code: str = None,
        assessment_title: str = None,
        grade_boundaries: Dict = None,
        progress_callback: Callable = None,
    ) -> Dict[str, Any]:
        """
        Process an entire assessment from a single root folder.

        Course code, course name, and assessment title are ALL auto-detected:
          - From a metadata file (course_info.txt / exam_info.json) in the folder.
          - Falling back to the folder name (e.g. CS101_Midterm → code=CS101, title=Midterm).
        Any resolved values can be overridden by passing explicit arguments.

        The folder PDFs are auto-classified into:
          - Answer key  (filename contains 'answer_key', 'key', 'solution' etc.)
          - Question paper (optional)
          - Student sheets (all remaining PDFs)

        Marks per question are auto-detected from the answer key by Gemini AI.

        Args:
            root_folder: Path to the exam folder.
            course_code: Override auto-detected course code.
            assessment_title: Override auto-detected assessment title.
            grade_boundaries: Optional grade boundaries.
            progress_callback: Optional callback(message).

        Returns:
            Summary dict from process().
        """
        def _progress(msg):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # ── Auto-detect course details ────────────────────────────────
        meta = parse_exam_metadata(root_folder)
        resolved_code = course_code or meta.course_code
        resolved_title = assessment_title or meta.assessment_title

        _progress(
            f"Course detected: {resolved_code} — [{resolved_title}] "
            f"(from {meta.source})"
        )

        # ── Auto-create course in DB if it doesn't exist ─────────────
        try:
            course = self.db.create_course(
                name=meta.course_name or resolved_code,
                code=resolved_code,
                department=meta.department,
            )
            _progress(f"Course ready: {resolved_code}")
        except Exception as e:
            logger.warning(f"Could not create/get course {resolved_code}: {e}")

        # ── Scan folder for PDFs ─────────────────────────────────────
        _progress(f"Scanning folder: {root_folder}")
        scan = scan_exam_folder(root_folder)
        _progress(describe_scan(scan))

        # Hard-stop if critical files are missing
        if not scan.answer_key_path:
            return {
                "course_code": resolved_code,
                "assessment": resolved_title,
                "students_processed": 0,
                "students_total": 0,
                "errors": scan.errors,
            }

        if not scan.student_paths:
            return {
                "course_code": resolved_code,
                "assessment": resolved_title,
                "students_processed": 0,
                "students_total": 0,
                "errors": scan.errors,
            }

        # Copy students into a unique temp dir so process() can glob for *.pdf
        import shutil
        import threading
        unique_suffix = f"{Path(root_folder).name}_{threading.get_ident()}"
        students_tmp = config.TEMP_DIR / f"students_{unique_suffix}"
        if students_tmp.exists():
            shutil.rmtree(students_tmp)
        students_tmp.mkdir(parents=True)
        for sp in scan.student_paths:
            shutil.copy2(sp, students_tmp / Path(sp).name)

        return self.process(
            course_code=resolved_code,
            assessment_title=resolved_title,
            answer_key_path=scan.answer_key_path,
            students_dir=str(students_tmp),
            marks_per_question=None,   # auto-detected from answer key or question paper
            grade_boundaries=grade_boundaries,
            progress_callback=progress_callback,
            question_paper_path=scan.question_paper_path,
        )

    def _process_answer_key(self, answer_key_path: str):
        """Process answer key PDF into per-question texts, images, and auto-detected marks."""
        answer_key_texts = {}
        answer_key_images = {}
        auto_marks = []

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

            # Combine all OCR text for mark extraction
            combined_ocr = "\n".join(ocr_texts)

            # Map to questions
            question_map = map_regions_to_questions(all_regions, ocr_texts)
            question_images_merged = merge_question_regions(question_map)

            for q_num, img in question_images_merged.items():
                answer_key_images[q_num] = img
                # OCR the merged image for text
                ocr_result = route_ocr(img)
                answer_key_texts[q_num] = ocr_result["text"]

            logger.info(f"Answer key: {len(answer_key_texts)} questions extracted")

            # Auto-detect marks per question using Gemini + regex
            question_count = len(answer_key_texts) or None
            auto_marks = parse_marks_from_answer_key(
                combined_ocr, question_count=question_count
            )
            logger.info(f"Auto-detected marks: {auto_marks}")

        except Exception as e:
            logger.error(f"Failed to process answer key: {e}")

        return answer_key_texts, answer_key_images, auto_marks

    def _extract_roll_number(self, filename: str) -> str:
        """Extract roll number from filename."""
        import re
        # Try patterns like: "123_StudentName", "ROLL123", "2024CS101"
        match = re.search(r"(\d{4,}[A-Z]*\d*)", filename)
        if match:
            return match.group(1)
        # Fallback: use filename
        return filename.replace(" ", "_")

    def _extract_marks_from_pdf(self, pdf_path: str) -> List[float]:
        """Extract marks per question directly from a PDF (e.g., question paper)."""
        try:
            images = pdf_to_images(pdf_path)
            ocr_texts = []
            for page_img in images:
                enhanced = enhance_scan(page_img)
                ocr_result = route_ocr(enhanced)
                ocr_texts.append(ocr_result["text"])
            
            combined_ocr = "\n".join(ocr_texts)
            return parse_marks_from_answer_key(combined_ocr)
        except Exception as e:
            logger.error(f"Failed to extract marks from PDF {pdf_path}: {e}")
            return []
