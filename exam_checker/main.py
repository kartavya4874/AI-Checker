"""
CLI entry point for the Hybrid AI Exam Checker.
Supports: --gui, --portal, --process modes.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid AI University Exam Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --gui                          Launch desktop GUI
  python main.py --portal                       Start web portal on localhost:8000
  python main.py --process --course "CS101" --assessment "Midterm" --answer-key key.pdf --students ./papers/
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--gui", action="store_true", help="Launch Tkinter desktop GUI")
    mode_group.add_argument("--portal", action="store_true", help="Start FastAPI web portal")
    mode_group.add_argument("--process", action="store_true", help="Process exams via CLI")

    # CLI processing arguments
    parser.add_argument("--course", type=str, help="Course code (for --process mode)")
    parser.add_argument("--assessment", type=str, help="Assessment title (for --process mode)")
    parser.add_argument("--answer-key", type=str, help="Path to answer key PDF")
    parser.add_argument("--students", type=str, help="Path to student PDFs directory")
    parser.add_argument("--samples", type=str, help="Path to teacher-marked sample PDFs (optional)")
    parser.add_argument("--marks", type=str, help="Comma-separated marks per question, e.g. '10,5,10,15'")

    args = parser.parse_args()

    from config import Config
    config = Config()

    if args.gui:
        _launch_gui(config)
    elif args.portal:
        _launch_portal(config)
    elif args.process:
        _run_processing(args, config)


def _launch_gui(config):
    """Launch the Tkinter desktop GUI."""
    print("Launching Exam Checker GUI...")
    try:
        from gui.main_window import ExamCheckerGUI
        app = ExamCheckerGUI()
        app.mainloop()
    except ImportError as e:
        print(f"Error: Could not launch GUI. {e}")
        sys.exit(1)


def _launch_portal(config):
    """Start the FastAPI web portal."""
    print(f"Starting web portal at http://{config.PORTAL_HOST}:{config.PORTAL_PORT}")
    try:
        import uvicorn
        from portal.app import create_app

        app = create_app()
        uvicorn.run(app, host=config.PORTAL_HOST, port=config.PORTAL_PORT)
    except ImportError as e:
        print(f"Error: Could not start portal. {e}")
        sys.exit(1)


def _run_processing(args, config):
    """Run CLI batch processing."""
    if not args.course or not args.assessment or not args.answer_key or not args.students:
        print("Error: --process requires --course, --assessment, --answer-key, and --students")
        sys.exit(1)

    config.validate()

    from pathlib import Path
    from database.db_manager import DatabaseManager
    from processing.course_processor import CourseProcessor

    # Parse marks allocation
    marks_per_question = None
    if args.marks:
        marks_per_question = [float(m.strip()) for m in args.marks.split(",")]

    answer_key_path = Path(args.answer_key)
    students_dir = Path(args.students)
    samples_dir = Path(args.samples) if args.samples else None

    if not answer_key_path.exists():
        print(f"Error: Answer key not found: {answer_key_path}")
        sys.exit(1)
    if not students_dir.exists():
        print(f"Error: Students directory not found: {students_dir}")
        sys.exit(1)

    # Initialize database
    db = DatabaseManager()
    db.init_db()

    # Run course processor
    processor = CourseProcessor(db)
    processor.process(
        course_code=args.course,
        assessment_title=args.assessment,
        answer_key_path=str(answer_key_path),
        students_dir=str(students_dir),
        marks_per_question=marks_per_question,
        samples_dir=str(samples_dir) if samples_dir else None,
        progress_callback=lambda msg: print(f"  {msg}"),
    )

    print("\n✓ Processing complete. Results saved to database.")
    print(f"  View results: python main.py --portal")


if __name__ == "__main__":
    main()
