"""
SQLAlchemy ORM models for the exam checker database.
"""

import json
from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, ForeignKey, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    code = Column(String(50), nullable=False, unique=True)
    department = Column(String(200), default="")
    semester = Column(String(50), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    assessments = relationship("Assessment", back_populates="course", cascade="all, delete-orphan")
    students = relationship("Student", back_populates="course", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Course {self.code}: {self.name}>"


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    total_marks = Column(Float, nullable=False, default=100.0)
    answer_key_path = Column(Text, default="")
    grade_boundaries = Column(JSON, default=None)
    marks_per_question = Column(JSON, default=None)  # List of marks per Q
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="assessments")
    results = relationship("Result", back_populates="assessment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Assessment {self.title} ({self.course_id})>"


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    roll_number = Column(String(50), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    email = Column(String(200), default="")

    # Relationships
    course = relationship("Course", back_populates="students")
    results = relationship("Result", back_populates="student", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Student {self.roll_number}: {self.name}>"


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    total_marks_obtained = Column(Float, default=0.0)
    total_marks_allocated = Column(Float, default=0.0)
    percentage = Column(Float, default=0.0)
    grade = Column(String(10), default="")
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, error
    processed_at = Column(DateTime, default=None)
    pdf_path = Column(Text, default="")

    # Relationships
    student = relationship("Student", back_populates="results")
    assessment = relationship("Assessment", back_populates="results")
    question_results = relationship(
        "QuestionResult", back_populates="result", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Result {self.student_id} - {self.assessment_id}: {self.percentage}%>"


class QuestionResult(Base):
    __tablename__ = "question_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    question_number = Column(Integer, nullable=False)
    marks_allocated = Column(Float, default=0.0)
    marks_obtained = Column(Float, default=0.0)
    content_type = Column(String(50), default="text")
    status = Column(String(50), default="unattempted")  # attempted, unattempted, partial
    feedback = Column(Text, default="")
    error_analysis = Column(Text, default="")
    partial_credit_breakdown = Column(JSON, default=None)
    ocr_text = Column(Text, default="")
    ocr_engine = Column(String(100), default="")
    pre_analysis = Column(JSON, default=None)

    # Relationships
    result = relationship("Result", back_populates="question_results")

    def __repr__(self):
        return f"<QuestionResult Q{self.question_number}: {self.marks_obtained}/{self.marks_allocated}>"


def init_database(database_url: str):
    """Initialize database and create all tables."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine):
    """Create a session factory."""
    return sessionmaker(bind=engine)
