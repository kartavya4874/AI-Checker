"""
PIL / numpy image utility helpers.
"""

import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter


def load_image(path: str) -> Image.Image:
    """Load an image from file path."""
    return Image.open(path).convert("RGB")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (BGR for OpenCV)."""
    arr = np.array(img)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return arr[:, :, ::-1]  # RGB → BGR
    return arr


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array (BGR) to PIL Image (RGB)."""
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(arr)


def resize_image(img: Image.Image, max_size: int = 2048) -> Image.Image:
    """Resize image so its longest side is at most max_size."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    ratio = max_size / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    return img.resize(new_size, Image.LANCZOS)


def crop_region(img: Image.Image, bbox: tuple) -> Image.Image:
    """Crop a region from image. bbox = (x1, y1, x2, y2)."""
    return img.crop(bbox)


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def enhance_contrast(img: Image.Image, factor: float = 1.5) -> Image.Image:
    """Enhance image contrast."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def enhance_sharpness(img: Image.Image, factor: float = 2.0) -> Image.Image:
    """Enhance image sharpness."""
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def save_image(img: Image.Image, path: str, fmt: str = None):
    """Save PIL Image to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt)
