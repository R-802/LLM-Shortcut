"""Classify prompt complexity into routing tiers."""

from __future__ import annotations

import re
from typing import Literal

Tier = Literal["fast", "balanced", "reasoning"]

_REASONING_MARKERS = re.compile(
    r"\b("
    r"analy[sz]e|prove|compare\s+tradeoffs?|architecture|"
    r"step\s+by\s+step|synthesi[sz]e|evaluate|derive|"
    r"justify|critically|trade[- ]?off"
    r")\b",
    re.IGNORECASE,
)

_CODE_BLOCK = re.compile(r"```[\s\S]{200,}```")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _context_section_size(prompt: str) -> int:
    marker = "Context Files:"
    idx = prompt.find(marker)
    if idx < 0:
        return 0
    return len(prompt[idx:])


def has_reasoning_markers(prompt: str) -> bool:
    if _REASONING_MARKERS.search(prompt):
        return True
    if _CODE_BLOCK.search(prompt):
        return True
    if _context_section_size(prompt) > 8000:
        return True
    return False


def classify(prompt: str, *, has_images: bool = False) -> Tier:
    tokens = estimate_tokens(prompt)
    markers = has_reasoning_markers(prompt)

    if tokens > 4000 or (markers and tokens > 1500):
        tier: Tier = "reasoning"
    elif tokens >= 400 or markers:
        tier = "balanced"
    else:
        tier = "fast"

    if has_images and tier == "fast":
        return "balanced"
    return tier


def next_tier(tier: Tier) -> Tier | None:
    order: list[Tier] = ["fast", "balanced", "reasoning"]
    idx = order.index(tier)
    if idx + 1 < len(order):
        return order[idx + 1]
    return None
