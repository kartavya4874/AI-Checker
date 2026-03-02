"""
Answer Key Parser: Auto-extract question count and marks-per-question
from an answer key using OpenAI GPT-4o with regex fallback.
"""

import re
import json
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger("answer_key_parser")

# Regex patterns to detect per-question marks in OCR text.
# Each pattern captures a question identifier and its mark value.
MARK_PATTERNS = [
    # "Q1. ... [5 marks]" or "Q1. ... [5 Marks]"
    r"(?:^|\n)\s*(?:Q|Que|Question|Ques|Ans|Answer|A)?[.\s:#-]*(\d+)[^[\n]*\[(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?|M)\]",
    # "Q1. ... (5 marks)" or "(5 pts)"
    r"(?:^|\n)\s*(?:Q|Que|Question|Ques|Ans|Answer|A)?[.\s:#-]*(\d+)[^(\n]*\((\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?|M)\)",
    # "Q1. ... 5 Marks" at end of line
    r"(?:^|\n)\s*(?:Q|Que|Question|Ques|Ans|Answer|A)?[.\s:#-]*(\d+)[^\n]*?(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?)\s*$",
    # "Q1 [Max: 5]" or "Q1 Max Marks: 5"
    r"(?:^|\n)\s*(?:Q|Que|Question|Ques)?[.\s:#-]*(\d+)[^\n]*?(?:max|maximum)[^\n]*?:\s*(\d+(?:\.\d+)?)",
    # Inline: "1. Answer... — 5 marks"
    r"(?:^|\n)\s*(\d+)\s*[.)]\s[^\n]*?—\s*(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?)",
]

# Pattern to detect a question boundary (finds question number only)
QUESTION_BOUNDARY_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Q|Que|Question|Ques|Ans|Answer)[.\s:#-]*(\d+)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_marks_from_answer_key(
    ocr_text: str,
    question_count: int = None,
    use_gemini: bool = True,   # kept for backward compat — internally uses OpenAI
) -> List[float]:
    """
    Auto-extract marks per question from the answer key OCR text.

    Strategy:
      1. Try OpenAI GPT-4o for structured extraction (most accurate).
      2. Fall back to regex pattern matching.
      3. If all else fails, return equal distribution of marks.

    Args:
        ocr_text: Combined OCR text from the answer key PDF.
        question_count: Known number of questions (optional hint).
        use_gemini: Legacy parameter — now triggers OpenAI extraction.

    Returns:
        List of marks per question (e.g. [10.0, 5.0, 15.0, ...]).
    """
    if not ocr_text or not ocr_text.strip():
        logger.warning("Empty OCR text — cannot extract marks  ")
        count = question_count or 1
        return [10.0] * count

    marks_list = None

    # --- Strategy 1: OpenAI GPT-4o ---
    if use_gemini:  # parameter kept for compat; now calls OpenAI
        try:
            marks_list = _extract_with_openai(ocr_text, question_count)
            if marks_list:
                logger.info(f"OpenAI extracted marks: {marks_list}")
        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {e}")

    # --- Strategy 2: Regex fallback ---
    if not marks_list:
        marks_list = extract_marks_from_text(ocr_text)
        if marks_list:
            logger.info(f"Regex extracted marks: {marks_list}")

    # --- Strategy 3: Equal distribution fallback ---
    if not marks_list:
        count = question_count or _count_questions_from_text(ocr_text) or 5
        marks_list = [10.0] * count
        logger.info(f"Using equal marks fallback: {marks_list}")

    # If question_count is known but marks list is shorter, pad it
    if question_count and len(marks_list) < question_count:
        avg = marks_list[-1] if marks_list else 10.0
        marks_list += [avg] * (question_count - len(marks_list))

    return marks_list


def extract_marks_from_text(text: str) -> List[float]:
    """
    Pure regex extraction of per-question marks from OCR text.
    Returns a sorted list of marks by question number, or [] if nothing found.
    """
    marks_by_question: Dict[int, float] = {}

    for pattern in MARK_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            try:
                q_num = int(match.group(1))
                mark = float(match.group(2))
                if 1 <= q_num <= 200 and 0 < mark <= 200:
                    if q_num not in marks_by_question:
                        marks_by_question[q_num] = mark
            except (ValueError, IndexError):
                continue

    if not marks_by_question:
        return []

    max_q = max(marks_by_question.keys())
    result = []
    prev = 10.0
    for i in range(1, max_q + 1):
        val = marks_by_question.get(i, prev)
        result.append(val)
        prev = val

    return result


def _count_questions_from_text(text: str) -> int:
    """Count distinct question numbers detected in text."""
    found = set()
    for m in QUESTION_BOUNDARY_PATTERN.finditer(text):
        try:
            q = int(m.group(1))
            if 1 <= q <= 200:
                found.add(q)
        except ValueError:
            pass
    return len(found)


def _extract_with_openai(ocr_text: str, question_count: int = None) -> List[float]:
    """
    Ask OpenAI GPT-4o to extract question count and marks per question as JSON.
    Returns a list of marks ordered by question number, or [] on failure.
    """
    try:
        from openai import OpenAI
        from config import config

        if not config.OPENAI_API_KEY:
            return []

        hint = f" There are {question_count} questions." if question_count else ""

        prompt = f"""You are an expert at reading university exam answer keys.
Analyse the following OCR text from an answer key and extract the marks allocated to EACH question.{hint}

Return a VALID JSON object in exactly this format:
{{
  "questions": [
    {{"number": 1, "marks": 10}},
    {{"number": 2, "marks": 5}},
    ...
  ]
}}

Rules:
- Include every question that has marks mentioned.
- If a question's marks are not explicitly stated, infer from context or use the most common mark value.
- Marks must be positive numbers.
- Do NOT include any text outside the JSON.

OCR TEXT:
---
{ocr_text[:4000]}
---
"""

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_EVAL_MODEL,
            messages=[
                {"role": "system", "content": "You extract structured data from text. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        questions = data.get("questions", [])

        if not questions:
            return []

        questions.sort(key=lambda q: q.get("number", 0))
        return [float(q["marks"]) for q in questions if "marks" in q]

    except Exception as e:
        logger.warning(f"OpenAI parse error: {e}")
        return []
