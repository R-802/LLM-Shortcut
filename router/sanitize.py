"""Normalise model output to allowed ASCII maths (+ - * / and x^n)."""

from __future__ import annotations

import re

# Disallowed symbols -> ASCII allowed forms
_SYMBOL_MAP = (
    ("π", "pi"),
    ("ρ", "rho"),
    ("η", "eta"),
    ("×", "*"),
    ("·", "*"),
    ("÷", "/"),
    ("√", "sqrt("),  # caller may leave trailing; rare in exam answers
    ("∑", "sum("),
    ("∏", "prod("),
    ("∫", "integral("),
    ("°", " deg"),
    ("≈", "~"),
    ("≤", "<="),
    ("≥", ">="),
    ("≠", "!="),
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
)

# Unicode superscripts -> ^n
_SUPERSCRIPT_MAP = {
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
}


def _unicode_superscripts_to_caret(text: str) -> str:
    for uni, caret in _SUPERSCRIPT_MAP.items():
        text = text.replace(uni, caret)
    return text


def sanitize_answer(text: str) -> str:
    if not text:
        return text

    text = _unicode_superscripts_to_caret(text)

    for old, new in _SYMBOL_MAP:
        text = text.replace(old, new)

    # Normalise caret spacing: "r ^ 2" -> "r^2"
    text = re.sub(r"\s*\^\s*(\d+)", r"^\1", text)

    # Collapse repeated spaces (keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()
