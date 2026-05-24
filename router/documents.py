"""Extract plain text from study materials for indexing and context."""

from __future__ import annotations

import re
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


def _row_looks_like_timestamp(cell: object) -> bool:
    s = str(cell).strip()
    if not s:
        return False
    if re.match(r"20\d{2}[-/]\d", s):
        return True
    return ":" in s and len(s) >= 10 and any(ch.isdigit() for ch in s)


def _extract_xlsx(file_path: Path) -> str:
    """Extract xlsx for RAG; skip huge hourly time-series sheets."""
    try:
        wb = load_workbook(file_path, data_only=True)
    except Exception:
        return ""

    parts: list[str] = []
    max_rows = 120

    for sheet in wb.worksheets:
        rows: list[list[str]] = []
        for row in sheet.iter_rows(values_only=True):
            rows.append([str(c) if c is not None else "" for c in row])

        if not rows:
            continue

        ts_rows = sum(1 for r in rows if r and _row_looks_like_timestamp(r[0]))
        if len(rows) > 30 and ts_rows >= max(20, int(len(rows) * 0.35)):
            parts.append(
                f"Sheet: {sheet.title} "
                f"(hourly time-series; {len(rows)} rows — summary only, not indexed row-by-row)"
            )
            continue

        parts.append(f"Sheet: {sheet.title}")
        for row in rows[:max_rows]:
            if any(cell.strip() for cell in row):
                parts.append(" | ".join(row))
        if len(rows) > max_rows:
            parts.append(f"... ({len(rows) - max_rows} more rows omitted)")

    return "\n".join(parts)


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(file_path)
    if suffix in {".xlsx", ".xls"}:
        return _extract_xlsx(file_path)
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
