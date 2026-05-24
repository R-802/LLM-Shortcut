"""Match user questions to context/ filenames and enrich chunks for embedding."""

from __future__ import annotations

import re
from pathlib import Path

_WORD_NUM: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
}


def _parse_num_token(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUM.get(token)


def _tutorial_pattern(num: int) -> re.Pattern[str]:
    # Tutorial_1 / Tutorial 1 / Tutorial-1 but not Tutorial_10 when num=1
    return re.compile(
        rf"tutorial[_\s-]*{num}(?:\.|_|[^\d]|$)",
        re.IGNORECASE,
    )


def _assignment_pattern(num: int) -> re.Pattern[str]:
    return re.compile(
        rf"assignment[_\s-]*{num}(?:\.|_|[^\d]|$)",
        re.IGNORECASE,
    )


def filename_tags(rel_path: str) -> str:
    """Human-readable tags duplicated in embedded text."""
    name = Path(rel_path).name
    low = name.lower()
    tags: list[str] = [f"file {name}"]

    for label, pattern in (
        ("tutorial", re.compile(r"tutorial[_\s-]*(\d+)", re.I)),
        ("assignment", re.compile(r"assignment[_\s-]*(\d+)", re.I)),
    ):
        m = pattern.search(low)
        if m:
            n = m.group(1)
            tags.append(f"{label} {n}")
            tags.append(f"{label} number {n}")

    if "lab" in low and "solar" in low:
        tags.append("solar photovoltaics lab")
    if "tutorial" in low:
        tags.append("RESE321 tutorial material")

    return " | ".join(tags)


def prepare_chunk_for_embedding(rel_path: str, chunk: str) -> str:
    """Prefix chunk so embeddings and retrieval see the document name."""
    name = Path(rel_path).name
    tags = filename_tags(rel_path)
    return (
        f"Document: {name}\n"
        f"Source path: {rel_path}\n"
        f"Keywords: {tags}\n\n"
        f"{chunk.strip()}"
    )


def infer_source_filters(question: str, known_sources: list[str]) -> list[str]:
    """
    Map phrases like 'tutorial one' or 'assignment 1' to context/ relative paths.
    Returns empty list if no specific document intent detected.
    """
    q = question.lower()
    matched: list[str] = []

    for m in re.finditer(r"\b(?:tutorial|tute|tut)\s*([a-z0-9]+)\b", q):
        num = _parse_num_token(m.group(1))
        if num is None:
            continue
        pat = _tutorial_pattern(num)
        for src in known_sources:
            if pat.search(src.replace(" ", "_")):
                matched.append(src)

    for m in re.finditer(r"\bassignment\s*([a-z0-9]+)\b", q):
        num = _parse_num_token(m.group(1))
        if num is None:
            continue
        pat = _assignment_pattern(num)
        for src in known_sources:
            if pat.search(src.replace(" ", "_")):
                matched.append(src)

    # "solar lab" / photovoltaics lab
    if re.search(r"\b(?:solar|pv)\s+(?:lab|photovoltaic)", q) or "photovoltaics lab" in q:
        for src in known_sources:
            if "lab" in src.lower() and "solar" in src.lower():
                matched.append(src)

    # Deduplicate, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for src in matched:
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out


def expand_query_with_sources(question: str, sources: list[str]) -> str:
    if not sources:
        return question
    names = ", ".join(Path(s).name for s in sources)
    return f"{question.strip()}\n\nFocus on these documents: {names}"
