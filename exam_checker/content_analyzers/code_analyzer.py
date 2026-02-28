"""
Code Analyzer: Language detection → static analysis → sandbox execution → GPT-4o review.
"""

import json
import re
import sys
import io
from PIL import Image
from typing import Dict, Any
from openai import OpenAI
from config import config
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("code_analyzer")

# Try importing RestrictedPython
try:
    from RestrictedPython import compile_restricted, safe_globals
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logger.warning("RestrictedPython not available — sandbox execution disabled")


def analyze_code(
    student_code: str,
    reference_code: str,
    marks_allocated: float,
    language: str = None,
) -> Dict[str, Any]:
    """
    Analyze student code against reference.

    Args:
        student_code: Student's code text.
        reference_code: Reference/expected code.
        marks_allocated: Total marks for this question.
        language: Programming language (auto-detected if None).

    Returns:
        Analysis result dict.
    """
    result = {
        "language": language or _detect_language(student_code),
        "syntax_valid": False,
        "static_analysis": [],
        "execution_result": None,
        "gpt4o_review": {},
        "suggested_marks": 0.0,
        "confidence": 0.0,
        "method": "code_analyzer",
    }

    try:
        lang = result["language"]

        # Step 1: Static analysis (Python only)
        if lang == "python":
            result["syntax_valid"] = _check_python_syntax(student_code)
            result["static_analysis"] = _python_static_analysis(student_code)

            # Step 2: Sandbox execution (Python only)
            if result["syntax_valid"] and RESTRICTED_PYTHON_AVAILABLE:
                exec_result = _sandbox_execute(student_code)
                result["execution_result"] = exec_result

        # Step 3: GPT-4o code review
        review = _gpt4o_code_review(student_code, reference_code, lang, marks_allocated)
        result["gpt4o_review"] = review

        # Calculate suggested marks from GPT-4o review
        if review and "suggested_marks" in review:
            result["suggested_marks"] = min(float(review["suggested_marks"]), marks_allocated)
            result["confidence"] = 0.8
        else:
            result["confidence"] = 0.5

    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        result["confidence"] = 0.2

    return result


def _detect_language(code: str) -> str:
    """Detect programming language from code."""
    code_lower = code.lower()

    # Python indicators
    if any(kw in code_lower for kw in ["def ", "import ", "print(", "elif ", "self."]):
        return "python"
    if re.search(r"^\s*#.*python", code_lower, re.MULTILINE):
        return "python"

    # Java indicators
    if any(kw in code_lower for kw in ["public static void", "system.out", "class ", "import java"]):
        return "java"

    # C/C++ indicators
    if any(kw in code_lower for kw in ["#include", "printf(", "int main(", "cout "]):
        return "c/c++"

    # JavaScript indicators
    if any(kw in code_lower for kw in ["console.log", "function ", "const ", "let ", "var "]):
        return "javascript"

    return "unknown"


def _check_python_syntax(code: str) -> bool:
    """Check Python syntax validity using compile()."""
    try:
        compile(code, "<student_code>", "exec")
        return True
    except SyntaxError as e:
        logger.debug(f"Syntax error: {e}")
        return False


def _python_static_analysis(code: str) -> list:
    """Basic Python static analysis."""
    issues = []

    # Check for common issues
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("import os") or stripped.startswith("import sys"):
            issues.append(f"Line {i}: System module import (potentially risky)")
        if "eval(" in stripped or "exec(" in stripped:
            issues.append(f"Line {i}: Use of eval/exec")
        if "open(" in stripped:
            issues.append(f"Line {i}: File I/O operation")

    # Check for undefined variables (basic)
    if "NameError" in code:
        issues.append("Possible undefined variable reference")

    return issues


def _sandbox_execute(code: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Execute Python code in RestrictedPython sandbox."""
    try:
        byte_code = compile_restricted(code, filename="<student>", mode="exec")
        if byte_code is None:
            return {"success": False, "error": "Compilation failed in restricted mode"}

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        restricted_globals = safe_globals.copy()
        restricted_globals["__builtins__"] = safe_globals["__builtins__"]
        restricted_globals["_print_"] = lambda *args, **kwargs: print(*args, **kwargs)
        restricted_globals["_getattr_"] = getattr

        try:
            exec(byte_code, restricted_globals)
            output = sys.stdout.getvalue()
            return {"success": True, "output": output[:1000]}
        except Exception as e:
            return {"success": False, "error": str(e)[:500]}
        finally:
            sys.stdout = old_stdout

    except Exception as e:
        return {"success": False, "error": f"Sandbox error: {str(e)[:500]}"}


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def _gpt4o_code_review(
    student_code: str,
    reference_code: str,
    language: str,
    marks_allocated: float,
) -> Dict[str, Any]:
    """Use GPT-4o to review student code against reference."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=config.OPENAI_EVAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a code review expert. Return ONLY valid JSON.",
            },
            {
                "role": "user",
                "content": (
                    f"Evaluate student code vs reference. Language: {language}\n"
                    f"Marks available: {marks_allocated}\n\n"
                    f"--- Student Code ---\n{student_code}\n\n"
                    f"--- Reference Code ---\n{reference_code}\n\n"
                    'Return JSON with: "logic_correct" (bool), "syntax_errors" (list), '
                    '"alternative_valid_approach" (bool), "suggested_marks" (number), '
                    '"feedback" (string).'
                ),
            },
        ],
        max_tokens=600,
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse code review JSON: {text[:200]}")
        return {"feedback": text, "suggested_marks": 0}
