"""
FastAPI web portal for the Hybrid AI Exam Checker.
"""

from pathlib import Path
from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import config
from database.db_manager import DatabaseManager
from processing.course_processor import CourseProcessor
from utils.folder_scanner import scan_exam_folder

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


def create_app() -> FastAPI:
    """Application factory used by CLI and ASGI servers."""
    return app


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
        
    course = db.get_course(assessment["course_id"])
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
        
    student = db.get_student(result["student_id"])
    assessment = db.get_assessment(result["assessment_id"])
    course = db.get_course(student["course_id"])
    
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
    exam_folders: List[str] = Form(...),
):
    """Start the exam checking pipeline from one or more folder paths.

    Course code, course name, and assessment title are all auto-detected
    from each folder name or a course_info.txt file inside it.
    All folders are processed in parallel.
    """
    # Validate all folders first before starting anything
    validated_folders = []
    for folder in exam_folders:
        folder_path = Path(folder.strip())
        if not folder_path.exists() or not folder_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

        # Quick scan to validate contents
        scan = scan_exam_folder(str(folder_path))
        if not scan.answer_key_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No answer key PDF found in folder '{folder_path.name}'. "
                    "Rename a file to include 'answer_key', 'key', or 'solution'."
                ),
            )
        if not scan.student_paths:
            raise HTTPException(
                status_code=400,
                detail=f"No student answer sheets found in folder '{folder_path.name}'.",
            )
        validated_folders.append(str(folder_path))

    # All folders valid — launch parallel background processing
    background_tasks.add_task(run_processing_task, validated_folders)
    return RedirectResponse(url="/", status_code=303)


def run_processing_task(exam_folders: List[str]):
    """Background task: process multiple exam folders in parallel."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = config.MAX_WORKERS

    def _process_one_folder(folder_path: str):
        """Process a single folder with its own CourseProcessor."""
        try:
            processor = CourseProcessor(db)
            processor.process_from_folder(root_folder=folder_path)
        except Exception as e:
            import logging
            logging.getLogger("portal").error(f"Failed to process folder {folder_path}: {e}")

    if len(exam_folders) == 1:
        # Single folder — no need for extra threading layer
        _process_one_folder(exam_folders[0])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one_folder, folder): folder
                for folder in exam_folders
            }
            for future in as_completed(futures):
                folder = futures[future]
                try:
                    future.result()
                except Exception as e:
                    import logging
                    logging.getLogger("portal").error(f"Thread error processing {folder}: {e}")


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
