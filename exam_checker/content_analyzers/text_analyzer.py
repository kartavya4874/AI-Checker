"""
Text Analyzer: OpenAI Embeddings cosine similarity + GPT-4o few-shot evaluation.
"""

import json
import numpy as np
from typing import Dict, Any, List
from openai import OpenAI
from config import config
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("text_analyzer")


def analyze_text_answer(
    student_text: str,
    answer_key_text: str,
    marks_allocated: float,
    few_shot_examples: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Analyze a text-based answer using embeddings + GPT-4o.

    Args:
        student_text: Student's answer text (from OCR).
        answer_key_text: Expected answer text.
        marks_allocated: Total marks for this question.
        few_shot_examples: Optional teacher-marked examples for few-shot.

    Returns:
        Analysis result dict.
    """
    result = {
        "similarity_score": 0.0,
        "gpt4o_evaluation": {},
        "suggested_marks": 0.0,
        "confidence": 0.0,
        "method": "text_analyzer",
    }

    try:
        # Step 1: Compute embedding similarity
        if student_text.strip() and answer_key_text.strip():
            similarity = _compute_similarity(student_text, answer_key_text)
            result["similarity_score"] = similarity
        else:
            similarity = 0.0

        # Step 2: GPT-4o evaluation with few-shot
        evaluation = _gpt4o_evaluate_text(
            student_text, answer_key_text, marks_allocated,
            similarity, few_shot_examples
        )
        result["gpt4o_evaluation"] = evaluation

        if evaluation and "marks_obtained" in evaluation:
            result["suggested_marks"] = min(
                float(evaluation["marks_obtained"]), marks_allocated
            )
            result["confidence"] = 0.85
        else:
            result["suggested_marks"] = round(marks_allocated * similarity, 1)
            result["confidence"] = 0.6

    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        result["confidence"] = 0.2

    return result


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def _compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity using OpenAI embeddings."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Get embeddings for both texts
    response = client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=[text1[:8000], text2[:8000]],
    )

    emb1 = np.array(response.data[0].embedding)
    emb2 = np.array(response.data[1].embedding)

    # Cosine similarity
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = float(dot / (norm1 * norm2))
    logger.debug(f"Embedding similarity: {similarity:.4f}")
    return max(0.0, min(1.0, similarity))


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def _gpt4o_evaluate_text(
    student_text: str,
    answer_key_text: str,
    marks_allocated: float,
    similarity_score: float,
    few_shot_examples: List[Dict] = None,
) -> Dict[str, Any]:
    """Use GPT-4o to evaluate text answer."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    messages = [
        {
            "role": "system",
            "content": "You are an expert examiner. Return ONLY valid JSON.",
        }
    ]

    # Add few-shot examples if available
    if few_shot_examples:
        for ex in few_shot_examples[:3]:
            messages.append({
                "role": "user",
                "content": (
                    f"Student answer: {ex.get('student_answer', '')}\n"
                    f"Answer key: {ex.get('answer_key', '')}\n"
                    f"Marks: {ex.get('marks', 0)}"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "marks_obtained": ex.get("marks_given", 0),
                    "feedback": ex.get("feedback", ""),
                    "key_points_covered": ex.get("points_covered", []),
                    "key_points_missing": ex.get("points_missing", []),
                }),
            })

    # Add actual evaluation request
    messages.append({
        "role": "user",
        "content": (
            f"Evaluate student answer vs answer key.\n"
            f"Marks available: {marks_allocated}\n"
            f"Semantic similarity: {similarity_score:.3f}\n\n"
            f"--- Student Answer ---\n{student_text[:3000]}\n\n"
            f"--- Answer Key ---\n{answer_key_text[:3000]}\n\n"
            'Return JSON: "marks_obtained" (number), "feedback" (string), '
            '"key_points_covered" (list), "key_points_missing" (list).'
        ),
    })

    response = client.chat.completions.create(
        model=config.OPENAI_EVAL_MODEL,
        messages=messages,
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
        logger.warning(f"Failed to parse text evaluation JSON: {text[:200]}")
        return {"feedback": text, "marks_obtained": 0}
