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


def _lab_pattern(num: int) -> re.Pattern[str]:
    return re.compile(
        rf"lab[_\s-]*{num}(?:\.|_|[^\d]|$)",
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


_EMBED_HEADER_RE = re.compile(
    r"^Document: .+\r?\nSource path: .+\r?\nKeywords: .+\r?\n\r?\n",
    re.DOTALL,
)


def chunk_body_for_prompt(stored_text: str) -> str:
    """Strip embedding-only metadata; return passage text for the LLM prompt."""
    text = (stored_text or "").strip()
    if not text:
        return ""
    m = _EMBED_HEADER_RE.match(text)
    if m:
        return text[m.end() :].strip()
    if text.startswith("Document:"):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return text


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

    for m in re.finditer(r"\blab\s*([a-z0-9]+)\b", q):
        num = _parse_num_token(m.group(1))
        if num is None:
            continue
        pat = _lab_pattern(num)
        for src in known_sources:
            if pat.search(src.replace(" ", "_")):
                matched.append(src)

    for m in re.finditer(r"\b([a-z]{4}\d{3})\b", q, re.IGNORECASE):
        code = m.group(1).lower()
        for src in known_sources:
            if code in _normalize_name(src):
                matched.append(src)

    for m in re.finditer(r"\bmath\s*(\d{3})\b", q, re.IGNORECASE):
        num = m.group(1)
        for src in known_sources:
            norm = _normalize_name(src)
            if f"math{num}" in norm or f"math_{num}" in norm:
                matched.append(src)

    # "solar lab" / photovoltaics lab
    if re.search(r"\b(?:solar|pv)\s+(?:lab|photovoltaic)", q) or "photovoltaics lab" in q:
        for src in known_sources:
            if "lab" in src.lower() and "solar" in src.lower():
                matched.append(src)

    # Filename keywords only when no tutorial/assignment/lab number was matched
    # (avoids "tutorial 2" also pulling Tutorial_1, _3, _4 via the word "tutorial")
    if not matched:
        for src in known_sources:
            if _filename_keyword_overlap(Path(src).name, question):
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


def _normalize_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def _shared_doc_id(filename_a: str, filename_b: str) -> bool:
    """True if both names refer to the same tutorial/assignment number."""
    a = _normalize_name(filename_a)
    b = _normalize_name(filename_b)
    for pattern in (
        re.compile(r"tutorial[_\s-]*(\d+)", re.I),
        re.compile(r"assignment[_\s-]*(\d+)", re.I),
    ):
        ma, mb = pattern.search(a), pattern.search(b)
        if ma and mb and ma.group(1) == mb.group(1):
            return True
    return False


_GENERIC_FILENAME_TOKENS = frozenset({
    "tutorial", "tute", "tut", "assignment", "assign", "lab",
    "report", "final", "sheet", "part", "solar", "photovoltaic",
    "rese321", "math132", "aiml339", "eeen301",
})


def _filename_keyword_overlap(filename: str, question: str) -> bool:
    """Match question words against the filename stem (skip generic/course tokens)."""
    stem = Path(filename).stem.lower().replace("_", " ").replace("-", " ")
    q = question.lower()

    if re.search(
        r"\b(?:tutorial|tute|tut|assignment|assign|lab)\s*[a-z0-9]+\b", q
    ):
        return False

    tokens = [
        t
        for t in re.split(r"\W+", stem)
        if len(t) >= 3 and t not in _GENERIC_FILENAME_TOKENS
    ]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in q)
    if hits == 0:
        return False
    if stem in q or q in stem:
        return True
    return hits >= 2 or (hits / len(tokens)) >= 0.5


def image_matches_question(
    filename: str,
    question: str,
    text_filters: list[str],
) -> bool:
    """
    Decide if a context/ image should be sent to the model.

    Uses the same tutorial/assignment/lab rules as text RAG, plus filename keywords.
    """
    name = Path(filename).name
    norm = _normalize_name(name)

    # Direct filename mention in filters (image was in known_sources list)
    if name in text_filters or filename in text_filters:
        return True

    if text_filters:
        for filt in text_filters:
            filt_name = Path(filt).name
            if norm == _normalize_name(filt_name):
                return True
            if _shared_doc_id(name, filt_name):
                return True
        return False

    # No named document intent — only include if filename keywords match the question
    if infer_source_filters(question, [name]):
        return True
    return _filename_keyword_overlap(name, question)


def list_context_image_names(source_dir: Path) -> list[str]:
    from router.images import IMAGE_SUFFIXES

    if not source_dir.is_dir():
        return []
    names: list[str] = []
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            names.append(path.name)
    return names
