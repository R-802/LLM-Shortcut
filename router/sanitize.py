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


_UNIT_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bmegawatts electrical\b", "MW"),
    (r"\bmegawatt electrical\b", "MW"),
    (r"\bmegawatts\b", "MW"),
    (r"\bmegawatt\b", "MW"),
    (r"\bkilowatts\b", "kW"),
    (r"\bkilowatt\b", "kW"),
    (r"\bkilojoules per kilogram\b", "kJ/kg"),
    (r"\bkilojoule per kilogram\b", "kJ/kg"),
    (r"\bkilojoules per hour\b", "kJ/h"),
    (r"\bkilojoule per hour\b", "kJ/h"),
    (r"\bkilojoules\b", "kJ"),
    (r"\bkilojoule\b", "kJ"),
    (r"\bdegrees celsius\b", "deg C"),
    (r"\bdegree celsius\b", "deg C"),
    (r"\btonnes per hour\b", "t/h"),
    (r"\btonne per hour\b", "t/h"),
    (r"\bwatts per kilowatt\b", "W/kW"),
    (r"\bwatts per megawatt\b", "W/MW"),
    (r"\bkilowatts per megawatt\b", "kW/MW"),
    (r"\bkilograms per tonne\b", "kg/t"),
    (r"\bseconds per hour\b", "s/h"),
)

_UNIT_SYMBOL_FIXES: tuple[tuple[str, str], ...] = (
    (r"\bMw\b", "MW"),
    (r"\bkw\b", "kW"),
    (r"\bKwh\b", "kWh"),
)


def _compact_units(text: str) -> str:
    for pattern, replacement in _UNIT_PHRASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for pattern, replacement in _UNIT_SYMBOL_FIXES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+percent\b", r"\1%", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+per\s+cent\b", r"\1%", text, flags=re.IGNORECASE)
    return text


def sanitize_answer(text: str) -> str:
    if not text:
        return text

    text = _unicode_superscripts_to_caret(text)

    for old, new in _SYMBOL_MAP:
        text = text.replace(old, new)

    text = _compact_units(text)

    # Normalise caret spacing: "r ^ 2" -> "r^2"
    text = re.sub(r"\s*\^\s*(\d+)", r"^\1", text)

    # Collapse repeated spaces (keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()
