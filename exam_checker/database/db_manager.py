"""
Database Manager: CRUD operations, session management, query helpers.
"""

import json
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from database.models import (
    Base, Course, Assessment, Student, Result, QuestionResult
)
from config import config
from utils.logger import get_logger

logger = get_logger("db_manager")


class DatabaseManager:
    """Database manager with CRUD operations and query helpers."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or config.DATABASE_URL
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionFactory = sessionmaker(bind=self.engine)

    def init_db(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {self.database_url}")

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # --- Course CRUD ---

    def create_course(self, name: str, code: str, department: str = "", semester: str = "") -> dict:
        with self.session_scope() as session:
            existing = session.query(Course).filter_by(code=code).first()
            if existing:
                return {"id": existing.id, "name": existing.name, "code": existing.code,
                        "department": existing.department, "semester": existing.semester}
            course = Course(name=name, code=code, department=department, semester=semester)
            session.add(course)
            session.flush()
            return {"id": course.id, "name": course.name, "code": course.code,
                    "department": course.department, "semester": course.semester}

    def get_course(self, course_id: int) -> Optional[dict]:
        with self.session_scope() as session:
            c = session.query(Course).get(course_id)
            if c is None:
                return None
            return {"id": c.id, "name": c.name, "code": c.code,
                    "department": c.department, "semester": c.semester}

    def get_course_by_code(self, code: str) -> Optional[dict]:
        with self.session_scope() as session:
            c = session.query(Course).filter_by(code=code).first()
            if c is None:
                return None
            return {"id": c.id, "name": c.name, "code": c.code,
                    "department": c.department, "semester": c.semester}

    def get_all_courses(self) -> List[Dict]:
        with self.session_scope() as session:
            courses = session.query(Course).all()
            return [
                {
                    "id": c.id, "name": c.name, "code": c.code,
                    "department": c.department, "semester": c.semester,
                    "created_at": str(c.created_at),
                    "student_count": session.query(Student).filter_by(course_id=c.id).count(),
                    "assessment_count": session.query(Assessment).filter_by(course_id=c.id).count(),
                }
                for c in courses
            ]

    def delete_course(self, course_id: int):
        with self.session_scope() as session:
            course = session.query(Course).get(course_id)
            if course:
                session.delete(course)

    # --- Assessment CRUD ---

    def create_assessment(
        self, course_id: int, title: str, total_marks: float,
        answer_key_path: str = "", marks_per_question: List[float] = None,
        grade_boundaries: Dict = None,
    ) -> dict:
        with self.session_scope() as session:
            assessment = Assessment(
                course_id=course_id,
                title=title,
                total_marks=total_marks,
                answer_key_path=answer_key_path,
                marks_per_question=marks_per_question,
                grade_boundaries=grade_boundaries,
            )
            session.add(assessment)
            session.flush()
            return {"id": assessment.id, "course_id": assessment.course_id,
                    "title": assessment.title, "total_marks": assessment.total_marks,
                    "answer_key_path": assessment.answer_key_path,
                    "marks_per_question": assessment.marks_per_question,
                    "grade_boundaries": assessment.grade_boundaries}

    def get_assessment(self, assessment_id: int) -> Optional[dict]:
        with self.session_scope() as session:
            a = session.query(Assessment).get(assessment_id)
            if a is None:
                return None
            return {"id": a.id, "course_id": a.course_id, "title": a.title,
                    "total_marks": a.total_marks, "answer_key_path": a.answer_key_path,
                    "marks_per_question": a.marks_per_question,
                    "grade_boundaries": a.grade_boundaries}

    def get_assessments_by_course(self, course_id: int) -> List[Dict]:
        with self.session_scope() as session:
            assessments = session.query(Assessment).filter_by(course_id=course_id).all()
            return [
                {
                    "id": a.id, "title": a.title, "total_marks": a.total_marks,
                    "answer_key_path": a.answer_key_path,
                    "marks_per_question": a.marks_per_question,
                    "created_at": str(a.created_at),
                    "result_count": session.query(Result).filter_by(assessment_id=a.id).count(),
                }
                for a in assessments
            ]

    # --- Student CRUD ---

    def create_student(self, name: str, roll_number: str, course_id: int, email: str = "") -> dict:
        with self.session_scope() as session:
            existing = session.query(Student).filter_by(
                roll_number=roll_number, course_id=course_id
            ).first()
            if existing:
                return {"id": existing.id, "name": existing.name,
                        "roll_number": existing.roll_number, "course_id": existing.course_id,
                        "email": existing.email}
            student = Student(name=name, roll_number=roll_number, course_id=course_id, email=email)
            session.add(student)
            session.flush()
            return {"id": student.id, "name": student.name,
                    "roll_number": student.roll_number, "course_id": student.course_id,
                    "email": student.email}

    def get_student(self, student_id: int) -> Optional[dict]:
        with self.session_scope() as session:
            s = session.query(Student).get(student_id)
            if s is None:
                return None
            return {"id": s.id, "name": s.name, "roll_number": s.roll_number,
                    "course_id": s.course_id, "email": s.email}

    def get_students_by_course(self, course_id: int) -> List[Dict]:
        with self.session_scope() as session:
            students = session.query(Student).filter_by(course_id=course_id).all()
            return [
                {"id": s.id, "name": s.name, "roll_number": s.roll_number, "email": s.email}
                for s in students
            ]

    # --- Result CRUD ---

    def create_result(
        self, student_id: int, assessment_id: int, pdf_path: str = ""
    ) -> dict:
        with self.session_scope() as session:
            result = Result(
                student_id=student_id,
                assessment_id=assessment_id,
                pdf_path=pdf_path,
                processing_status="pending",
            )
            session.add(result)
            session.flush()
            return {"id": result.id, "student_id": result.student_id,
                    "assessment_id": result.assessment_id, "pdf_path": result.pdf_path,
                    "processing_status": result.processing_status}

    def get_result(self, result_id: int) -> Optional[dict]:
        with self.session_scope() as session:
            r = session.query(Result).get(result_id)
            if r is None:
                return None
            return {"id": r.id, "student_id": r.student_id,
                    "assessment_id": r.assessment_id, "pdf_path": r.pdf_path,
                    "total_marks_obtained": r.total_marks_obtained,
                    "total_marks_allocated": r.total_marks_allocated,
                    "percentage": r.percentage, "grade": r.grade,
                    "processing_status": r.processing_status}

    def update_result(
        self, result_id: int,
        total_marks_obtained: float = None,
        total_marks_allocated: float = None,
        percentage: float = None,
        grade: str = None,
        processing_status: str = None,
    ):
        with self.session_scope() as session:
            result = session.query(Result).get(result_id)
            if result:
                if total_marks_obtained is not None:
                    result.total_marks_obtained = total_marks_obtained
                if total_marks_allocated is not None:
                    result.total_marks_allocated = total_marks_allocated
                if percentage is not None:
                    result.percentage = percentage
                if grade is not None:
                    result.grade = grade
                if processing_status is not None:
                    result.processing_status = processing_status
                if processing_status == "completed":
                    result.processed_at = datetime.utcnow()

    def get_results_by_assessment(self, assessment_id: int) -> List[Dict]:
        with self.session_scope() as session:
            results = (
                session.query(Result, Student)
                .join(Student, Result.student_id == Student.id)
                .filter(Result.assessment_id == assessment_id)
                .all()
            )
            return [
                {
                    "id": r.id, "student_id": r.student_id,
                    "student_name": s.name, "roll_number": s.roll_number,
                    "total_marks_obtained": r.total_marks_obtained,
                    "total_marks_allocated": r.total_marks_allocated,
                    "percentage": r.percentage, "grade": r.grade,
                    "processing_status": r.processing_status,
                    "processed_at": str(r.processed_at) if r.processed_at else "",
                    "pdf_path": r.pdf_path,
                }
                for r, s in results
            ]

    def get_student_results(self, student_id: int) -> List[Dict]:
        with self.session_scope() as session:
            results = (
                session.query(Result, Assessment)
                .join(Assessment, Result.assessment_id == Assessment.id)
                .filter(Result.student_id == student_id)
                .all()
            )
            return [
                {
                    "id": r.id, "assessment_id": r.assessment_id,
                    "assessment_title": a.title,
                    "total_marks_obtained": r.total_marks_obtained,
                    "total_marks_allocated": r.total_marks_allocated,
                    "percentage": r.percentage, "grade": r.grade,
                    "processing_status": r.processing_status,
                }
                for r, a in results
            ]

    # --- QuestionResult CRUD ---

    def save_question_result(
        self, result_id: int, question_data: Dict[str, Any]
    ) -> QuestionResult:
        with self.session_scope() as session:
            qr = QuestionResult(
                result_id=result_id,
                question_number=question_data.get("question_number", 0),
                marks_allocated=question_data.get("marks_allocated", 0),
                marks_obtained=question_data.get("marks_obtained", 0),
                content_type=question_data.get("content_type", "text"),
                status=question_data.get("status", "unattempted"),
                feedback=question_data.get("feedback", ""),
                error_analysis=question_data.get("error_analysis", ""),
                partial_credit_breakdown=question_data.get("partial_credit_breakdown"),
                ocr_text=question_data.get("ocr_text", ""),
                ocr_engine=question_data.get("ocr_engine", ""),
                pre_analysis=question_data.get("pre_analysis"),
            )
            session.add(qr)
            session.flush()
            qr_id = qr.id
        return qr_id

    def get_question_results(self, result_id: int) -> List[Dict]:
        with self.session_scope() as session:
            qrs = (
                session.query(QuestionResult)
                .filter_by(result_id=result_id)
                .order_by(QuestionResult.question_number)
                .all()
            )
            return [
                {
                    "id": qr.id,
                    "question_number": qr.question_number,
                    "marks_allocated": qr.marks_allocated,
                    "marks_obtained": qr.marks_obtained,
                    "content_type": qr.content_type,
                    "status": qr.status,
                    "feedback": qr.feedback,
                    "error_analysis": qr.error_analysis,
                    "partial_credit_breakdown": qr.partial_credit_breakdown,
                    "ocr_text": qr.ocr_text,
                    "ocr_engine": qr.ocr_engine,
                    "pre_analysis": qr.pre_analysis,
                }
                for qr in qrs
            ]

    def get_all_question_results_for_assessment(self, assessment_id: int) -> List[Dict]:
        with self.session_scope() as session:
            qrs = (
                session.query(QuestionResult)
                .join(Result, QuestionResult.result_id == Result.id)
                .filter(Result.assessment_id == assessment_id)
                .all()
            )
            return [
                {
                    "question_number": qr.question_number,
                    "marks_allocated": qr.marks_allocated,
                    "marks_obtained": qr.marks_obtained,
                    "content_type": qr.content_type,
                    "status": qr.status,
                }
                for qr in qrs
            ]

    # --- Export ---

    def export_results_csv(self, assessment_id: int) -> str:
        """Export assessment results as CSV string."""
        results = self.get_results_by_assessment(assessment_id)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Roll Number", "Student Name", "Marks Obtained",
            "Total Marks", "Percentage", "Grade", "Status"
        ])
        for r in results:
            writer.writerow([
                r["roll_number"], r["student_name"],
                r["total_marks_obtained"], r["total_marks_allocated"],
                r["percentage"], r["grade"], r["processing_status"],
            ])
        return output.getvalue()

    def export_results_excel(self, assessment_id: int, output_path: str):
        """Export assessment results as Excel file."""
        try:
            from openpyxl import Workbook
            wb = Workbook()

            # Summary sheet
            ws = wb.active
            ws.title = "Results Summary"
            headers = [
                "Roll Number", "Student Name", "Marks Obtained",
                "Total Marks", "Percentage", "Grade", "Status"
            ]
            ws.append(headers)

            results = self.get_results_by_assessment(assessment_id)
            for r in results:
                ws.append([
                    r["roll_number"], r["student_name"],
                    r["total_marks_obtained"], r["total_marks_allocated"],
                    r["percentage"], r["grade"], r["processing_status"],
                ])

            # Question-wise sheet
            ws2 = wb.create_sheet("Question Analysis")
            ws2.append([
                "Roll Number", "Student", "Question", "Marks Allocated",
                "Marks Obtained", "Content Type", "Status", "Feedback"
            ])

            for r in results:
                qrs = self.get_question_results(r["id"])
                for qr in qrs:
                    ws2.append([
                        r["roll_number"], r["student_name"],
                        qr["question_number"], qr["marks_allocated"],
                        qr["marks_obtained"], qr["content_type"],
                        qr["status"], qr["feedback"][:200],
                    ])

            wb.save(output_path)
            logger.info(f"Exported results to {output_path}")

        except ImportError:
            logger.error("openpyxl not installed — cannot export Excel")
            raise

    # --- Statistics Helpers ---

    def get_assessment_stats(self, assessment_id: int) -> Dict[str, Any]:
        """Get assessment statistics."""
        results = self.get_results_by_assessment(assessment_id)
        if not results:
            return {"count": 0}

        percentages = [r["percentage"] for r in results if r["processing_status"] == "completed"]
        if not percentages:
            return {"count": len(results), "completed": 0}

        import math
        n = len(percentages)
        mean = sum(percentages) / n
        sorted_p = sorted(percentages)
        median = sorted_p[n // 2] if n % 2 else (sorted_p[n // 2 - 1] + sorted_p[n // 2]) / 2
        variance = sum((p - mean) ** 2 for p in percentages) / n
        std_dev = math.sqrt(variance)

        grades = [r["grade"] for r in results if r["grade"]]
        grade_dist = {}
        for g in grades:
            grade_dist[g] = grade_dist.get(g, 0) + 1

        pass_count = sum(1 for p in percentages if p >= 33)

        return {
            "count": len(results),
            "completed": n,
            "mean": round(mean, 2),
            "median": round(median, 2),
            "std_dev": round(std_dev, 2),
            "min": round(min(percentages), 2),
            "max": round(max(percentages), 2),
            "pass_rate": round((pass_count / n) * 100, 2) if n > 0 else 0,
            "grade_distribution": grade_dist,
        }
