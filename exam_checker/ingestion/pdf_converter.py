"""
PDF → list of PIL Images using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import List
from utils.logger import get_logger

logger = get_logger("pdf_converter")


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images (one per page).

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Default 300.

    Returns:
        List of PIL Image objects, one per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    images = []
    try:
        doc = fitz.open(str(pdf_path))
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                logger.debug(f"Page {page_num + 1}/{len(doc)} converted: {pix.width}x{pix.height}")
            except Exception as e:
                logger.error(f"Failed to convert page {page_num + 1}: {e}")
                continue

        doc.close()
        logger.info(f"Converted {len(images)} pages from {pdf_path.name}")

    except fitz.FileDataError:
        logger.error(f"Corrupt or password-protected PDF: {pdf_path}")
        raise ValueError(f"Cannot open PDF: {pdf_path}. File may be corrupt or password-protected.")
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise

    return images


def pdf_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF."""
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0
