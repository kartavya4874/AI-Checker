"""
Folder Scanner: Discover question paper, answer key, and student answer sheets
from a single root folder. The user provides one path; this module figures out
which file is which by inspecting filenames, and auto-extracts course details.

Expected folder layout (flat or with subfolders):

  CS101_Midterm/                ← folder name parsed → code=CS101, title=Midterm
    question_paper.pdf          ← any PDF with "question", "qp", "paper" in name
    answer_key.pdf              ← any PDF with "answer", "key", "solution" in name
    student_001.pdf             ← everything else = student sheets
    student_002.pdf

Optional metadata file inside the folder (overrides folder-name parsing):
  course_info.txt  OR  exam_info.json
    Course Code: CS101
    Course Name: Computer Science
    Assessment: Midterm Examination
    Department: Engineering

If a file's name matches BOTH answer-key and question-paper patterns, it is
treated as the answer key (answer key takes priority).
"""

import re
import json
from pathlib import Path
from typing import NamedTuple, List, Optional
from utils.logger import get_logger

logger = get_logger("folder_scanner")

# --------------------------------------------------------------------------- #
# Filename keyword patterns
# --------------------------------------------------------------------------- #

ANSWER_KEY_KEYWORDS = re.compile(
    r"answer[_\s-]?key|answerkey|answer[_\s-]?sheet[_\s-]?key|"
    r"\bkey\b|solution|solutions|ans[_\s-]?key|\bmarks\b",
    re.IGNORECASE,
)

QUESTION_PAPER_KEYWORDS = re.compile(
    r"question[_\s-]?paper|questionpaper|\bqp\b|question[_\s-]?bank|"
    r"\bpaper\b|exam[_\s-]?paper|test[_\s-]?paper|\bquestion\b",
    re.IGNORECASE,
)

# Subfolder names that commonly hold each file category
ANSWER_KEY_FOLDERS = {"answer_key", "answerkey", "answer", "key", "solution", "solutions"}
QUESTION_FOLDERS = {"question", "question_paper", "qp", "paper"}
STUDENT_FOLDERS = {"students", "student", "scripts", "answers", "submissions", "sheets"}


class FolderScanResult(NamedTuple):
    answer_key_path: Optional[str]      # Path to the answer key PDF
    question_paper_path: Optional[str]  # Path to the question paper PDF (optional)
    student_paths: List[str]            # Paths to all student answer sheet PDFs
    errors: List[str]                   # Validation errors / warnings


def scan_exam_folder(root_folder: str) -> FolderScanResult:
    """
    Scan a root exam folder and categorise all PDFs.

    Args:
        root_folder: Path to the root directory.

    Returns:
        FolderScanResult with paths and any error messages.
    """
    root = Path(root_folder)
    errors: List[str] = []

    if not root.exists():
        return FolderScanResult(None, None, [], [f"Folder not found: {root_folder}"])
    if not root.is_dir():
        return FolderScanResult(None, None, [], [f"Path is not a folder: {root_folder}"])

    # ------------------------------------------------------------------ #
    # Collect all PDFs recursively (but keep track of their parent folder)
    # ------------------------------------------------------------------ #
    all_pdfs = list(root.rglob("*.pdf"))
    if not all_pdfs:
        return FolderScanResult(None, None, [], ["No PDF files found in the folder"])

    answer_key: Optional[Path] = None
    question_paper: Optional[Path] = None
    student_sheets: List[Path] = []

    # ------------------------------------------------------------------ #
    # Pass 1: subfolder-based classification
    # ------------------------------------------------------------------ #
    unclassified: List[Path] = []
    for pdf in all_pdfs:
        folder_name = pdf.parent.name.lower().strip()
        if folder_name in ANSWER_KEY_FOLDERS:
            answer_key = _pick_one(answer_key, pdf, "answer key", errors)
        elif folder_name in QUESTION_FOLDERS:
            question_paper = _pick_one(question_paper, pdf, "question paper", errors)
        elif folder_name in STUDENT_FOLDERS or pdf.parent != root:
            if pdf.parent.name.lower() not in ANSWER_KEY_FOLDERS | QUESTION_FOLDERS:
                student_sheets.append(pdf)
        else:
            unclassified.append(pdf)

    # ------------------------------------------------------------------ #
    # Pass 2: filename-based classification for unclassified files
    # ------------------------------------------------------------------ #
    still_unclassified: List[Path] = []
    for pdf in unclassified:
        stem = pdf.stem
        is_key = bool(ANSWER_KEY_KEYWORDS.search(stem))
        is_qp = bool(QUESTION_PAPER_KEYWORDS.search(stem))

        if is_key:
            answer_key = _pick_one(answer_key, pdf, "answer key", errors)
        elif is_qp:
            question_paper = _pick_one(question_paper, pdf, "question paper", errors)
        else:
            still_unclassified.append(pdf)

    # Everything left is treated as student answer sheets
    student_sheets.extend(still_unclassified)
    student_sheets.sort(key=lambda p: p.name)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    if answer_key is None:
        errors.append(
            "No answer key found. Name a file with 'answer_key', 'key', or 'solution' "
            "in its filename, or place it in an 'answer_key/' subfolder."
        )
    if not student_sheets:
        errors.append(
            "No student answer sheets found. All PDFs were classified as answer key / question paper."
        )

    logger.info(
        f"Scan complete — answer_key={answer_key}, "
        f"question_paper={question_paper}, "
        f"students={len(student_sheets)}"
    )

    return FolderScanResult(
        answer_key_path=str(answer_key) if answer_key else None,
        question_paper_path=str(question_paper) if question_paper else None,
        student_paths=[str(p) for p in student_sheets],
        errors=errors,
    )


def _pick_one(
    existing: Optional[Path],
    candidate: Path,
    role: str,
    errors: List[str],
) -> Path:
    """Keep the first match; log a warning if a second one appears."""
    if existing is None:
        return candidate
    errors.append(
        f"Multiple {role} candidates found; using '{existing.name}', "
        f"ignoring '{candidate.name}'. Rename files to resolve ambiguity."
    )
    return existing


def describe_scan(result: "FolderScanResult") -> str:
    """Return a human-readable summary of the scan result."""
    lines = []
    lines.append(f"  Answer Key:     {result.answer_key_path or 'NOT FOUND'}")
    lines.append(f"  Question Paper: {result.question_paper_path or 'not provided (optional)'}")
    lines.append(f"  Student Sheets: {len(result.student_paths)} file(s)")
    if result.student_paths:
        for p in result.student_paths:
            lines.append(f"    · {Path(p).name}")
    if result.errors:
        lines.append("  Warnings/Errors:")
        for e in result.errors:
            lines.append(f"    ⚠ {e}")
    return "\n".join(lines)


# =========================================================================== #
# Course / Assessment Metadata                                                  #
# =========================================================================== #

class ExamMetadata(NamedTuple):
    """Auto-detected course and assessment information."""
    course_code: str        # e.g. "CS101"
    course_name: str        # e.g. "Computer Science"
    assessment_title: str   # e.g. "Midterm Examination"
    department: str         # e.g. "Engineering"  (may be empty)
    source: str             # "metadata_file" | "folder_name" | "default"


# Names of metadata files we look for inside the exam folder
_META_FILES = [
    "course_info.txt", "course_info.json",
    "exam_info.txt", "exam_info.json",
    "metadata.txt", "metadata.json",
    "info.txt", "info.json",
]

# Regex to split a folder name like "CS101_Midterm_2" into tokens
_FOLDER_SPLIT = re.compile(r"[\s_\-]+")

# Pattern that looks like a course code: 2–4 letters + optional digits (e.g. CS101, MATH3A)
_COURSE_CODE_RE = re.compile(r"^[A-Za-z]{2,6}\d*[A-Za-z]?\d*$")


def parse_exam_metadata(root_folder: str) -> ExamMetadata:
    """
    Auto-detect course code, course name, and assessment title from the folder.

    Detection order (highest priority first):
      1. A metadata file (course_info.txt / exam_info.json / ...) inside the folder.
      2. The folder name itself (e.g. ``CS101_Midterm`` → code=CS101, title=Midterm).
      3. Safe defaults.

    Metadata file format (key: value, one per line, .txt):
        Course Code: CS101
        Course Name: Computer Science
        Assessment:  Midterm Examination
        Department:  Engineering

    Or JSON (.json):
        {"course_code": "CS101", "course_name": "...", "assessment": "..."}

    Args:
        root_folder: Path to the exam root folder.

    Returns:
        ExamMetadata with detected values.
    """
    root = Path(root_folder)

    # ------------------------------------------------------------------ #
    # Strategy 1: metadata file                                            #
    # ------------------------------------------------------------------ #
    for fname in _META_FILES:
        meta_path = root / fname
        if meta_path.exists():
            try:
                return _parse_metadata_file(meta_path)
            except Exception as e:
                logger.warning(f"Could not parse metadata file {fname}: {e}")

    # ------------------------------------------------------------------ #
    # Strategy 2: folder name parsing                                      #
    # ------------------------------------------------------------------ #
    return _parse_from_folder_name(root)


def _parse_metadata_file(path: Path) -> ExamMetadata:
    """Parse a metadata file (txt or json) into ExamMetadata."""
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return ExamMetadata(
            course_code=str(data.get("course_code", data.get("code", "COURSE"))).strip().upper(),
            course_name=str(data.get("course_name", data.get("name", ""))).strip(),
            assessment_title=str(data.get("assessment", data.get("title", "Assessment"))).strip(),
            department=str(data.get("department", data.get("dept", ""))).strip(),
            source="metadata_file",
        )
    else:
        # Key: Value line format
        fields = {}
        for line in text.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                fields[key.strip().lower()] = val.strip()

        code = fields.get("course code", fields.get("code", "")).upper() or "COURSE"
        name = fields.get("course name", fields.get("name", ""))
        title = fields.get("assessment", fields.get("title", fields.get("exam", "Assessment")))
        dept = fields.get("department", fields.get("dept", ""))

        return ExamMetadata(
            course_code=code,
            course_name=name or code,
            assessment_title=title,
            department=dept,
            source="metadata_file",
        )


def _parse_from_folder_name(root: Path) -> ExamMetadata:
    """
    Infer course code and assessment title from the folder name.

    Examples:
      CS101_Midterm        → code=CS101, title=Midterm
      CS101_Midterm_2      → code=CS101, title=Midterm 2
      Math101 - Final Exam → code=Math101, title=Final Exam
      Midterm_CS101        → code=CS101, title=Midterm
      Final_Exam_2024      → code=COURSE, title=Final Exam 2024
    """
    folder_name = root.name.strip()
    tokens = [t for t in _FOLDER_SPLIT.split(folder_name) if t]

    course_code = None
    title_tokens = []

    for tok in tokens:
        if course_code is None and _COURSE_CODE_RE.match(tok) and len(tok) >= 3:
            course_code = tok.upper()   # take FIRST code-like token
        else:
            title_tokens.append(tok)

    if course_code is None:
        course_code = "COURSE"

    assessment_title = " ".join(title_tokens) if title_tokens else folder_name
    if not assessment_title:
        assessment_title = "Assessment"

    logger.info(f"Parsed folder name '{folder_name}' → code={course_code}, title={assessment_title}")

    return ExamMetadata(
        course_code=course_code,
        course_name=course_code,   # name defaults to code; user can update in DB
        assessment_title=assessment_title,
        department="",
        source="folder_name",
    )

