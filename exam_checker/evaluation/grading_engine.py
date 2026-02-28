"""
Grading Engine: Calculate grades from marks, course statistics, question-wise analysis.
"""

import json
import math
from typing import Dict, Any, List
from config import config
from utils.logger import get_logger

logger = get_logger("grading_engine")


def calculate_grade(
    marks_obtained: float,
    total_marks: float,
    grade_boundaries: Dict[str, float] = None,
) -> str:
    """
    Calculate grade based on percentage and grade boundaries.

    Args:
        marks_obtained: Total marks obtained.
        total_marks: Total marks available.
        grade_boundaries: Dict of grade → min percentage (e.g., {"A+": 90, "A": 80, ...}).

    Returns:
        Grade string (e.g., "A+", "B", "F").
    """
    if total_marks <= 0:
        return "N/A"

    percentage = (marks_obtained / total_marks) * 100
    boundaries = grade_boundaries or config.GRADE_BOUNDARIES

    # Sort boundaries by percentage descending
    sorted_grades = sorted(boundaries.items(), key=lambda x: x[1], reverse=True)

    for grade, min_pct in sorted_grades:
        if percentage >= min_pct:
            return grade

    return "F"


def calculate_percentage(marks_obtained: float, total_marks: float) -> float:
    """Calculate percentage rounded to 2 decimal places."""
    if total_marks <= 0:
        return 0.0
    return round((marks_obtained / total_marks) * 100, 2)


def apply_negative_marking(
    marks_obtained: float,
    incorrect_count: int,
    negative_factor: float = None,
) -> float:
    """Apply negative marking for incorrect answers."""
    factor = negative_factor if negative_factor is not None else config.NEGATIVE_MARKING_FACTOR
    penalty = incorrect_count * factor
    return max(0, marks_obtained - penalty)


def calculate_course_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive course statistics.

    Args:
        results: List of result dicts with marks_obtained, total_marks, etc.

    Returns:
        Statistics dict with mean, median, std_dev, pass_rate, etc.
    """
    if not results:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "std_dev": 0,
            "min_marks": 0,
            "max_marks": 0,
            "pass_rate": 0,
            "distinction_rate": 0,
            "grade_distribution": {},
        }

    percentages = []
    grades = []

    for r in results:
        total = r.get("total_marks", r.get("total_marks_allocated", 0))
        obtained = r.get("marks_obtained", r.get("total_marks_obtained", 0))
        pct = calculate_percentage(obtained, total)
        percentages.append(pct)
        grades.append(r.get("grade", calculate_grade(obtained, total)))

    n = len(percentages)
    mean = sum(percentages) / n
    sorted_pcts = sorted(percentages)

    # Median
    if n % 2 == 0:
        median = (sorted_pcts[n // 2 - 1] + sorted_pcts[n // 2]) / 2
    else:
        median = sorted_pcts[n // 2]

    # Standard deviation
    variance = sum((p - mean) ** 2 for p in percentages) / n
    std_dev = math.sqrt(variance)

    # Grade distribution
    grade_dist = {}
    for g in grades:
        grade_dist[g] = grade_dist.get(g, 0) + 1

    # Pass rate (>= 33%)
    pass_count = sum(1 for p in percentages if p >= 33)
    distinction_count = sum(1 for p in percentages if p >= 75)

    return {
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std_dev": round(std_dev, 2),
        "min_marks": round(min(percentages), 2),
        "max_marks": round(max(percentages), 2),
        "pass_rate": round((pass_count / n) * 100, 2),
        "distinction_rate": round((distinction_count / n) * 100, 2),
        "grade_distribution": grade_dist,
    }


def calculate_question_statistics(
    question_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Calculate per-question statistics across all students.

    Args:
        question_results: List of per-question result dicts.

    Returns:
        List of question-wise statistics.
    """
    # Group by question number
    question_groups: Dict[int, List] = {}
    for qr in question_results:
        q_num = qr.get("question_number", 0)
        if q_num not in question_groups:
            question_groups[q_num] = []
        question_groups[q_num].append(qr)

    stats = []
    for q_num in sorted(question_groups.keys()):
        qrs = question_groups[q_num]
        marks_obtained = [qr.get("marks_obtained", 0) for qr in qrs]
        marks_allocated = qrs[0].get("marks_allocated", 0) if qrs else 0

        n = len(marks_obtained)
        mean = sum(marks_obtained) / n if n > 0 else 0
        max_m = max(marks_obtained) if marks_obtained else 0
        min_m = min(marks_obtained) if marks_obtained else 0

        # Difficulty classification
        avg_ratio = mean / marks_allocated if marks_allocated > 0 else 0
        if avg_ratio >= 0.8:
            difficulty = "Easy"
        elif avg_ratio >= 0.5:
            difficulty = "Medium"
        elif avg_ratio >= 0.3:
            difficulty = "Hard"
        else:
            difficulty = "Very Hard"

        # Content type (most common)
        content_types = [qr.get("content_type", "text") for qr in qrs]
        most_common_type = max(set(content_types), key=content_types.count)

        # Attempt rate
        attempted = sum(1 for qr in qrs if qr.get("status") != "unattempted")
        attempt_rate = (attempted / n * 100) if n > 0 else 0

        stats.append({
            "question_number": q_num,
            "marks_allocated": marks_allocated,
            "mean_score": round(mean, 2),
            "max_score": max_m,
            "min_score": min_m,
            "difficulty": difficulty,
            "content_type": most_common_type,
            "attempt_rate": round(attempt_rate, 2),
            "student_count": n,
        })

    return stats
