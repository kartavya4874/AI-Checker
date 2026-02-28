"""
Chemistry Analyzer: GPT-4o Vision predicts SMILES → RDKit Tanimoto comparison.
Also: chemical equation parsing + balancing check.
"""

import re
import json
from PIL import Image
from typing import Dict, Any
from openai import OpenAI
from config import config
from utils.image_utils import image_to_base64
from utils.retry_utils import retry_with_backoff
from utils.logger import get_logger

logger = get_logger("chemistry_analyzer")

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available — SMILES comparison disabled")


def analyze_chemistry_answer(
    student_img: Image.Image,
    answer_key_text: str,
    marks_allocated: float,
    content_subtype: str = "structure",
) -> Dict[str, Any]:
    """
    Analyze a chemistry answer.

    Args:
        student_img: PIL Image of student's chemistry answer.
        answer_key_text: Expected answer (SMILES or equation).
        marks_allocated: Total marks for this question.
        content_subtype: "structure" or "equation".

    Returns:
        Analysis result dict.
    """
    if content_subtype == "structure":
        return analyze_structure(student_img, answer_key_text, marks_allocated)
    else:
        return analyze_equation(student_img, answer_key_text, marks_allocated)


def analyze_structure(
    student_img: Image.Image,
    answer_smiles: str,
    marks_allocated: float,
) -> Dict[str, Any]:
    """Analyze chemical structure via SMILES prediction + Tanimoto."""
    result = {
        "predicted_smiles": "",
        "answer_smiles": answer_smiles,
        "tanimoto_similarity": 0.0,
        "suggested_marks": 0.0,
        "confidence": 0.0,
        "method": "chemistry_structure",
    }

    try:
        # Step 1: Predict SMILES from image using GPT-4o
        predicted = _predict_smiles_gpt4o(student_img)
        result["predicted_smiles"] = predicted

        if not predicted:
            result["confidence"] = 0.3
            return result

        # Step 2: Compare using RDKit Tanimoto
        if RDKIT_AVAILABLE and answer_smiles:
            similarity = _tanimoto_similarity(predicted, answer_smiles)
            result["tanimoto_similarity"] = similarity
            result["suggested_marks"] = round(marks_allocated * similarity, 1)
            result["confidence"] = 0.8
        else:
            # Fallback: string comparison
            result["tanimoto_similarity"] = 1.0 if predicted.strip() == answer_smiles.strip() else 0.0
            result["suggested_marks"] = marks_allocated if result["tanimoto_similarity"] > 0.9 else 0.0
            result["confidence"] = 0.5

    except Exception as e:
        logger.error(f"Structure analysis failed: {e}")
        result["confidence"] = 0.2

    return result


def analyze_equation(
    student_img: Image.Image,
    answer_equation: str,
    marks_allocated: float,
) -> Dict[str, Any]:
    """Analyze chemical equation — parse and check balancing."""
    result = {
        "student_equation": "",
        "answer_equation": answer_equation,
        "is_balanced": False,
        "reactants_correct": False,
        "products_correct": False,
        "suggested_marks": 0.0,
        "confidence": 0.0,
        "method": "chemistry_equation",
    }

    try:
        # Extract equation from image using GPT-4o
        student_eq = _extract_equation_gpt4o(student_img)
        result["student_equation"] = student_eq

        if not student_eq:
            return result

        # Parse and compare
        student_parsed = _parse_equation(student_eq)
        answer_parsed = _parse_equation(answer_equation)

        if student_parsed and answer_parsed:
            result["reactants_correct"] = (
                student_parsed["reactants"] == answer_parsed["reactants"]
            )
            result["products_correct"] = (
                student_parsed["products"] == answer_parsed["products"]
            )
            result["is_balanced"] = _check_balanced(student_parsed)

            # Score
            score = 0.0
            if result["reactants_correct"]:
                score += 0.3
            if result["products_correct"]:
                score += 0.3
            if result["is_balanced"]:
                score += 0.4

            result["suggested_marks"] = round(marks_allocated * score, 1)
            result["confidence"] = 0.7

    except Exception as e:
        logger.error(f"Equation analysis failed: {e}")
        result["confidence"] = 0.2

    return result


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def _predict_smiles_gpt4o(img: Image.Image) -> str:
    """Use GPT-4o Vision to predict SMILES from structure image."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    b64 = image_to_base64(img)

    response = client.chat.completions.create(
        model=config.OPENAI_EVAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This image contains a hand-drawn chemical structure. "
                            "Predict the SMILES string. Return ONLY the SMILES string."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=200,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


@retry_with_backoff(max_retries=2, base_delay=2.0, rate_limit=True)
def _extract_equation_gpt4o(img: Image.Image) -> str:
    """Use GPT-4o to extract chemical equation from image."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    b64 = image_to_base64(img)

    response = client.chat.completions.create(
        model=config.OPENAI_EVAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract the chemical equation from this image. "
                            "Return ONLY the equation in standard notation "
                            "(e.g., 2H2 + O2 -> 2H2O). Use -> for the arrow."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=300,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


def _tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two SMILES strings using RDKit."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    except Exception as e:
        logger.debug(f"Tanimoto calculation failed: {e}")
        return 0.0


def _parse_equation(equation: str) -> Dict:
    """Parse a chemical equation into reactants and products."""
    try:
        # Split by arrow
        for sep in ["->", "→", "=>", "⟶"]:
            if sep in equation:
                parts = equation.split(sep)
                if len(parts) == 2:
                    reactants = _parse_side(parts[0])
                    products = _parse_side(parts[1])
                    return {"reactants": reactants, "products": products, "raw": equation}
        return None
    except Exception:
        return None


def _parse_side(side: str) -> Dict[str, int]:
    """Parse one side of a chemical equation into compound counts."""
    compounds = {}
    parts = re.split(r"\s*\+\s*", side.strip())
    for part in parts:
        part = part.strip()
        match = re.match(r"(\d*)\s*([A-Z].*)", part)
        if match:
            coeff = int(match.group(1)) if match.group(1) else 1
            formula = match.group(2).strip()
            compounds[formula] = compounds.get(formula, 0) + coeff
    return compounds


def _check_balanced(parsed: Dict) -> bool:
    """Check if a parsed equation is balanced by counting atoms."""
    try:
        left_atoms = _count_atoms_side(parsed["reactants"])
        right_atoms = _count_atoms_side(parsed["products"])
        return left_atoms == right_atoms
    except Exception:
        return False


def _count_atoms_side(compounds: Dict[str, int]) -> Dict[str, int]:
    """Count total atoms on one side of equation."""
    total = {}
    for formula, coeff in compounds.items():
        atoms = _parse_formula(formula)
        for atom, count in atoms.items():
            total[atom] = total.get(atom, 0) + count * coeff
    return total


def _parse_formula(formula: str) -> Dict[str, int]:
    """Parse chemical formula into atom counts (simple parser)."""
    atoms = {}
    pattern = r"([A-Z][a-z]?)(\d*)"
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        atoms[element] = atoms.get(element, 0) + count
    return atoms
