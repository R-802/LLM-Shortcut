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


_LECTURE_TOPIC_TAGS: dict[str, list[str]] = {
    "3-1": ["solar resource", "irradiance", "solarview", "solar radiation"],
    "3-2": ["solar collectors", "flat plate", "evacuated tube", "water heater", "SWH", "collector area"],
    "4-1": ["solar photovoltaics", "PV", "solar panel", "I-V curve", "MPP"],
    "4-2": ["PV systems", "battery", "inverter", "off-grid", "grid-connected"],
    "5-1": ["heat engines", "Rankine cycle", "steam", "thermal efficiency"],
    "6-1": ["power cycles", "ORC", "combined cycle"],
    "6-2": ["geothermal", "binary cycle", "geothermal fluid", "enthalpy"],
    "7-1": ["wind power", "wind turbine", "wind farm", "Cp", "Betz", "rotor", "wind speed"],
    "7-2": ["wind farm development", "wind resource"],
    "8-1": ["water power", "hydroelectric", "hydro", "dam", "head", "flow rate"],
    "9-1": ["bioenergy", "biomass", "wood fuel", "calorific value"],
    "9-2": ["biofuels", "biodiesel", "ethanol"],
    "9-3": ["hydrogen", "fuel cell", "electrolysis"],
    "10-1": ["energy storage", "battery storage", "pumped hydro"],
    "11-1": ["hybrid systems", "diesel hybrid"],
}

_SOLUTION_KEYWORDS = [
    "problem solution", "problem solutions", "worked answer",
    "worked example", "calculation", "answer sheet",
]


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

    lect_m = re.search(r"lecture\s*(\d+[-_]\d+)", low)
    if lect_m:
        key = lect_m.group(1).replace("_", "-")
        topic_tags = _LECTURE_TOPIC_TAGS.get(key, [])
        tags.extend(topic_tags)
        if any(kw in low for kw in ("problem", "solution", "answer")):
            tags.append("worked solution")
            tags.append("exam calculation")
            for t in topic_tags:
                if t not in tags:
                    tags.append(t)

    return " | ".join(tags)


def find_companion_solutions(
    retrieved_sources: list[str],
    known_sources: list[str],
) -> list[str]:
    """
    Given sources retrieved by vector search, return any companion solutions xlsx.
    E.g. 'Lecture 3-2 - Solar energy collectors.pdf'
    -> 'Lecture 3-2 - Solar energy collectors problem solutions.xlsx'
    """
    companions: list[str] = []
    seen: set[str] = set(retrieved_sources)

    for src in retrieved_sources:
        name = Path(src).name
        m = re.match(r"(lecture\s*\d+[-_]\d+)", name, re.IGNORECASE)
        if not m:
            continue
        prefix = m.group(1).lower()
        for candidate in known_sources:
            if candidate in seen:
                continue
            cname = Path(candidate).name.lower()
            if not cname.startswith(prefix):
                continue
            if candidate.lower().endswith((".xlsx", ".xls")):
                if any(kw in cname for kw in ("problem", "solution", "answer")):
                    companions.append(candidate)
                    seen.add(candidate)

    return companions


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


def _dedupe_sources(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for src in sources:
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out


def match_explicit_filenames(question: str, known_sources: list[str]) -> list[str]:
    """When the user names a file (e.g. 'use previous test 2 answers.xlsx'), match only that file."""
    q_lower = question.lower()
    q_compact = re.sub(r"[\s_()\-]+", "", q_lower)
    matched: list[str] = []
    for src in known_sources:
        name = Path(src).name.lower()
        if name in q_lower:
            matched.append(src)
            continue
        compact = re.sub(r"[\s_()\-]+", "", name)
        if len(compact) >= 10 and compact in q_compact:
            matched.append(src)
    return _dedupe_sources(matched)


def match_test_number_files(question: str, known_sources: list[str]) -> list[str]:
    """'previous test 2' -> Test 2 files only, not Test 1."""
    matched: list[str] = []
    for m in re.finditer(r"\b(?:previous\s+)?test\s*([0-9]+)\b", question, re.IGNORECASE):
        num = m.group(1)
        pat = re.compile(rf"test[_\s-]*{num}(?:\.|_|[^\d]|$)", re.IGNORECASE)
        for src in known_sources:
            if pat.search(_normalize_name(Path(src).name)):
                matched.append(src)
    return _dedupe_sources(matched)


def slice_xlsx_for_question(text: str, question: str) -> str:
    """Prefer one sheet block when the question names a sheet (e.g. 'question 2')."""
    if "=== Sheet:" not in text:
        return text
    m = re.search(r"\bquestion\s*(\d+)\b", question, re.IGNORECASE)
    if not m:
        return text
    label = f"=== Sheet: Question {m.group(1)} ==="
    start = text.find(label)
    if start < 0:
        return text
    rest = text[start + len(label) :]
    nxt = rest.find("=== Sheet:")
    block = label + (rest[:nxt] if nxt >= 0 else rest)
    return block.strip()


def infer_source_filters(question: str, known_sources: list[str]) -> list[str]:
    """
    Map phrases like 'tutorial one' or 'assignment 1' to context/ relative paths.
    Returns empty list if no specific document intent detected.
    """
    explicit = match_explicit_filenames(question, known_sources)
    if explicit:
        return explicit

    test_files = match_test_number_files(question, known_sources)
    if test_files:
        return test_files

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

    # Topic-keyword → lecture matching: catch specific terms from the question
    # that map directly to a particular lecture's subject matter.
    _TOPIC_LECTURE_MAP: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\b(?:flat plate|evacuated tube|swh|water heater|collector area)\b"), "3-2"),
        (re.compile(r"\b(?:solarview|irradiance|solar resource|tilted irradiance)\b"), "3-1"),
        (re.compile(r"\b(?:photovoltaic|pv panel|pv system|iv curve|mpp|inverter|off.grid)\b"), "4-"),
        (re.compile(r"\b(?:rankine|steam turbine|carnot|thermal efficiency|heat engine)\b"), "5-1"),
        (re.compile(r"\b(?:geothermal|binary cycle|geothermal fluid|enthalpy|ngawha)\b"), "6-2"),
        (re.compile(r"\b(?:wind turbine|wind farm|betz|rotor diameter|cut.in|makara)\b"), "7-1"),
        (re.compile(r"\b(?:hydroelectric|hydro|manapouri|penstock|turbine head|dam)\b"), "8-1"),
        (re.compile(r"\b(?:biomass|biofuel|calorific|biodiesel|ethanol|firewood|wood fuel)\b"), "9-"),
        (re.compile(r"\b(?:hydrogen|fuel cell|electrolysis)\b"), "9-3"),
        (re.compile(r"\b(?:battery storage|pumped hydro|flywheel|energy storage)\b"), "10-"),
    ]

    if not matched:
        for pattern, lect_key in _TOPIC_LECTURE_MAP:
            if pattern.search(q):
                for src in known_sources:
                    lect_m = re.search(r"lecture\s*(\d+[-_]\d+)", src, re.IGNORECASE)
                    if lect_m and lect_m.group(1).replace("_", "-").startswith(lect_key):
                        matched.append(src)

    # Filename keyword overlap (last resort, tighter thresholds than before)
    if not matched:
        for src in known_sources:
            if _filename_keyword_overlap(Path(src).name, question):
                matched.append(src)

    return _dedupe_sources(matched)


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
    # Course admin / structure
    "tutorial", "tute", "tut", "assignment", "assign", "lab",
    "report", "final", "sheet", "part",
    "previous", "test", "questions", "answers", "question",
    "problem", "problems", "solution", "solutions", "lecture",
    # Course codes
    "rese321", "math132", "aiml339", "eeen301",
    # Common English filler (appear in almost every question / filename)
    "and", "the", "for", "with", "from", "this", "that", "are",
    "energy", "power", "system", "systems",
    # Domain-generic (appear in many lectures and many questions)
    "solar", "photovoltaic", "heat", "hot",
})

# Minimum token length for keyword overlap matching (avoids short noise words)
_MIN_KW_TOKEN_LEN = 5


def _filename_keyword_overlap(filename: str, question: str) -> bool:
    """
    Match distinctive (>=5 char) words from the filename stem against the question.
    Returns True only when at least 2 non-generic words match, or the ratio is >=0.5.
    """
    stem = Path(filename).stem.lower().replace("_", " ").replace("-", " ")
    q = question.lower()

    if re.search(
        r"\b(?:tutorial|tute|tut|assignment|assign|lab)\s*[a-z0-9]+\b", q
    ):
        return False

    tokens = [
        t
        for t in re.split(r"\W+", stem)
        if len(t) >= _MIN_KW_TOKEN_LEN and t not in _GENERIC_FILENAME_TOKENS
    ]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in q)
    if hits == 0:
        return False
    if stem in q or q in stem:
        return True
    return hits >= 2 or (hits / len(tokens)) >= 0.6


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
