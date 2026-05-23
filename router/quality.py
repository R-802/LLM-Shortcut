"""Light quality checks — escalate only on clearly bad answers."""

from __future__ import annotations

import re

_REFUSAL = re.compile(
    r"^(as an ai|i cannot|i can't|i am unable|sorry,? i)",
    re.IGNORECASE,
)

_FAILURE_SNIPPETS = (
    "error:",
    "429",
    "rate limit",
    "quota exceeded",
    "resource exhausted",
    "all providers exhausted",
)


def is_weak(answer: str, question: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True

    lower = text.lower()
    for snippet in _FAILURE_SNIPPETS:
        if snippet in lower:
            return True

    if _REFUSAL.match(text):
        return True

    q = (question or "").strip()
    if len(q) > 40 and len(text) < 20:
        return True

    return False
