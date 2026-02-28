"""
FastAPI web portal for the Hybrid AI Exam Checker.
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import config
from database.db_manager import DatabaseManager
from processing.course_processor import CourseProcessor

# Setup
app = FastAPI(title="Hybrid AI Exam Checker Portal")

# Paths
PORTAL_DIR = Path(__file__).parent
TEMPLATES_DIR = PORTAL_DIR / "templates"
STATIC_DIR = PORTAL_DIR / "static"

# Create static dir if it doesn't exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Templates & Static
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Database
db = DatabaseManager()
db.init_db()


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard overview showing courses and stats."""
    courses = db.get_all_courses()
    
    total_students = sum(c["student_count"] for c in courses)
    total_assessments = sum(c["assessment_count"] for c in courses)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "courses": courses,
        "total_courses": len(courses),
        "total_students": total_students,
        "total_assessments": total_assessments,
    })


@app.get("/course/{course_id}", response_class=HTMLResponse)
async def course_detail(request: Request, course_id: int):
    """Course detail page showing assessments."""
    course = db.get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
        
    assessments = db.get_assessments_by_course(course_id)
    students = db.get_students_by_course(course_id)
    
    # Add stats to assessments
    for a in assessments:
        stats = db.get_assessment_stats(a["id"])
        a["stats"] = stats
        
    return templates.TemplateResponse("course_detail.html", {
        "request": request,
        "course": course,
        "assessments": assessments,
        "students": students,
    })


@app.get("/assessment/{assessment_id}", response_class=HTMLResponse)
async def assessment_detail(request: Request, assessment_id: int):
    """Assessment detail page showing results and stats."""
    assessment = db.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
        
    course = db.get_course(assessment.course_id)
    results = db.get_results_by_assessment(assessment_id)
    stats = db.get_assessment_stats(assessment_id)
    
    return templates.TemplateResponse("assessment_detail.html", {
        "request": request,
        "course": course,
        "assessment": assessment,
        "results": results,
        "stats": stats,
    })


@app.get("/student/{result_id}", response_class=HTMLResponse)
async def student_detail(request: Request, result_id: int):
    """Student result detail showing per-question breakdown."""
    result = db.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
        
    student = db.get_student(result.student_id)
    assessment = db.get_assessment(result.assessment_id)
    course = db.get_course(student.course_id)
    
    question_results = db.get_question_results(result_id)
    
    return templates.TemplateResponse("student_detail.html", {
        "request": request,
        "course": course,
        "assessment": assessment,
        "student": student,
        "result": result,
        "question_results": question_results,
    })


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page for processing new exams."""
    courses = db.get_all_courses()
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "courses": courses,
    })


@app.post("/start_processing")
async def start_processing(
    background_tasks: BackgroundTasks,
    course_id: int = Form(...),
    assessment_title: str = Form(...),
    marks_per_question: str = Form(...),
    answer_key: UploadFile = File(...),
    student_files: List[UploadFile] = File(...),
    sample_files: List[UploadFile] = File(None),
):
    """Start the exam checking pipeline."""
    # Setup temp dirs
    upload_dir = config.TEMP_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save answer key
    key_path = upload_dir / f"key_{answer_key.filename}"
    with open(key_path, "wb") as f:
        f.write(await answer_key.read())
        
    # Save student files
    student_dir = upload_dir / "students"
    student_dir.mkdir(exist_ok=True)
    for s_file in student_files:
        with open(student_dir / s_file.filename, "wb") as f:
            f.write(await s_file.read())
            
    # Save samples if any
    sample_dir = None
    if sample_files and sample_files[0].filename:
        sample_dir = upload_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        for sam_file in sample_files:
            with open(sample_dir / sam_file.filename, "wb") as f:
                f.write(await sam_file.read())
                
    # Parse marks
    try:
        marks_list = [float(m.strip()) for m in marks_per_question.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid marks format")
        
    # Get course details
    course = db.get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
        
    # Start task
    background_tasks.add_task(
        run_processing_task,
        course.code,
        assessment_title,
        str(key_path),
        str(student_dir),
        marks_list,
        str(sample_dir) if sample_dir else None
    )
    
    return RedirectResponse(url="/", status_code=303)


def run_processing_task(
    course_code: str,
    assessment_title: str,
    answer_key_path: str,
    students_dir: str,
    marks_per_question: List[float],
    samples_dir: Optional[str],
):
    """Background task to run course processor."""
    processor = CourseProcessor(db)
    processor.process(
        course_code=course_code,
        assessment_title=assessment_title,
        answer_key_path=answer_key_path,
        students_dir=students_dir,
        marks_per_question=marks_per_question,
        samples_dir=samples_dir,
    )


@app.get("/export/{assessment_id}/csv")
async def export_csv(assessment_id: int):
    """Export assessment results as CSV."""
    csv_data = db.export_results_csv(assessment_id)
    
    # Write to temp file
    temp_path = config.TEMP_DIR / f"export_{assessment_id}.csv"
    with open(temp_path, "w", newline="") as f:
        f.write(csv_data)
        
    return FileResponse(
        path=temp_path,
        filename=f"results_assessment_{assessment_id}.csv",
        media_type="text/csv"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("portal.app:app", host=config.PORTAL_HOST, port=config.PORTAL_PORT, reload=True)
