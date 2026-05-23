"""Extract plain text from study materials for indexing and context."""

from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook
from pypdf import PdfReader

# Files never indexed as RAG sources
SKIP_NAMES = {
    "app.py", "app.log", ".env", "requirements.txt",
    "ROUTER_TEST.md", ".env.example",
}

SKIP_SUFFIXES = {".pyc", ".yaml", ".bat", ".ps1", ".log"}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

INDEXABLE_SUFFIXES = {
    ".pdf", ".xlsx", ".xls", ".txt", ".md", ".csv", ".json", ".py",
    ".html", ".htm", ".docx",
}


def read_pdf(file_path: Path) -> str:
    text = ""
    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        text = "PDF could not be read."
    return text


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(file_path)
    if suffix in {".xlsx", ".xls"}:
        try:
            wb = load_workbook(file_path, data_only=True)
            parts: list[str] = []
            for sheet in wb.worksheets:
                parts.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    cells = [str(c) if c is not None else "" for c in row]
                    parts.append(" | ".join(cells))
            return "\n".join(parts)
        except Exception:
            return ""
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def iter_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.name.startswith("."):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES | IMAGE_SUFFIXES:
            continue
        if path.suffix.lower() not in INDEXABLE_SUFFIXES:
            continue
        files.append(path)
    return files
