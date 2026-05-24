"""Load and encode images for vision-capable LLM routes."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_IMAGES = 6
MAX_EDGE_PX = 1568
JPEG_QUALITY = 85


@dataclass
class ImageAttachment:
    label: str
    data_url: str


def _pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for images. Run: pip install Pillow"
        ) from exc
    return Image


def _encode_image(img) -> str:
    Image = _pil_image()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((MAX_EDGE_PX, MAX_EDGE_PX), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def attachment_from_path(path: Path, label: str | None = None) -> ImageAttachment | None:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    Image = _pil_image()
    try:
        with Image.open(path) as img:
            data_url = _encode_image(img.copy())
        name = label or path.name
        return ImageAttachment(label=name, data_url=data_url)
    except OSError as exc:
        logger.warning("Could not read image %s: %s", path, exc)
        return None


def attachment_from_clipboard(label: str = "clipboard") -> ImageAttachment | None:
    Image = _pil_image()
    try:
        from PIL import ImageGrab
    except ImportError:
        return None
    try:
        img = ImageGrab.grabclipboard()
    except OSError as exc:
        logger.warning("Clipboard image read failed: %s", exc)
        return None
    if img is None:
        return None
    # File path(s) on clipboard
    if isinstance(img, list):
        paths = [p for p in img if isinstance(p, str) and Path(p).is_file()]
        if paths:
            return attachment_from_path(Path(paths[0]), label)
        return None
    try:
        data_url = _encode_image(img)
        return ImageAttachment(label=label, data_url=data_url)
    except (OSError, ValueError) as exc:
        logger.warning("Could not encode clipboard image: %s", exc)
        return None


def dedupe_attachments(images: list[ImageAttachment]) -> list[ImageAttachment]:
    seen: set[str] = set()
    unique: list[ImageAttachment] = []
    for img in images:
        if img.data_url in seen:
            continue
        seen.add(img.data_url)
        unique.append(img)
    return unique


def collect_from_dirs(*folders: Path) -> list[ImageAttachment]:
    found: list[ImageAttachment] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            att = attachment_from_path(path)
            if att:
                found.append(att)
            if len(found) >= MAX_IMAGES:
                return found
    return found


def collect_matching_from_dir(
    folder: Path,
    question: str,
    text_filters: list[str],
) -> list[ImageAttachment]:
    """Load images from folder only when filename matches the question (see rag_sources)."""
    from router.rag_sources import image_matches_question

    if not folder.is_dir():
        return []

    found: list[ImageAttachment] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not image_matches_question(path.name, question, text_filters):
            continue
        att = attachment_from_path(path)
        if att:
            found.append(att)
        if len(found) >= MAX_IMAGES:
            break
    return found
