"""Parse and validate Windows RegisterHotKey shortcuts (e.g. ctrl+b)."""

from __future__ import annotations

import os
import re

DEFAULT_HOTKEY = "ctrl+b"

# RegisterHotKey modifier flags
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

MODIFIER_NAMES: dict[str, int] = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "windows": MOD_WIN,
}

# Optional blocklist for awkward system shortcuts
BLOCKED_COMBOS: frozenset[str] = frozenset()

FKEY_RE = re.compile(r"^f([1-9]|1[0-2])$", re.IGNORECASE)


def _vk_for_key(key: str) -> int | None:
    k = key.lower()
    if len(k) == 1 and k.isalpha():
        return ord(k.upper())
    if len(k) == 1 and k.isdigit():
        return ord(k)
    m = FKEY_RE.match(k)
    if m:
        return 0x6F + int(m.group(1))  # VK_F1 = 0x70
    return None


def validate_hotkey_string(raw: str) -> dict:
    """
    Return {ok, normalized, modifiers, vk, display, error}.
    normalized is lowercase like ctrl+shift+b.
    """
    text = (raw or "").strip().lower().replace(" ", "")
    if not text:
        text = DEFAULT_HOTKEY

    parts = [p for p in text.split("+") if p]
    if len(parts) < 2:
        return {
            "ok": False,
            "error": "Use at least one modifier and one key (e.g. ctrl+b).",
        }

    modifiers = 0
    key_part: str | None = None
    for part in parts:
        if part in MODIFIER_NAMES:
            modifiers |= MODIFIER_NAMES[part]
        elif key_part is None:
            key_part = part
        else:
            return {
                "ok": False,
                "error": "Use only one non-modifier key (e.g. ctrl+shift+f5).",
            }

    if modifiers == 0:
        return {
            "ok": False,
            "error": "Include a modifier: ctrl, alt, shift, or win.",
        }

    if key_part is None:
        return {
            "ok": False,
            "error": "Missing the key after modifiers (e.g. ctrl+b).",
        }

    vk = _vk_for_key(key_part)
    if vk is None:
        return {
            "ok": False,
            "error": "Key must be a letter, digit, or f1-f12.",
        }

    mod_names = []
    if modifiers & MOD_CONTROL:
        mod_names.append("ctrl")
    if modifiers & MOD_ALT:
        mod_names.append("alt")
    if modifiers & MOD_SHIFT:
        mod_names.append("shift")
    if modifiers & MOD_WIN:
        mod_names.append("win")
    normalized = "+".join(mod_names + [key_part])
    display = normalized.upper().replace("+", "+").replace("CTRL", "Ctrl")

    if normalized in BLOCKED_COMBOS:
        return {
            "ok": False,
            "error": "That shortcut is not allowed. Choose another.",
        }

    # Friendly display: Ctrl+B
    display_parts = []
    if modifiers & MOD_CONTROL:
        display_parts.append("Ctrl")
    if modifiers & MOD_ALT:
        display_parts.append("Alt")
    if modifiers & MOD_SHIFT:
        display_parts.append("Shift")
    if modifiers & MOD_WIN:
        display_parts.append("Win")
    display_parts.append(key_part.upper() if len(key_part) == 1 else key_part.upper())
    display = "+".join(display_parts)

    return {
        "ok": True,
        "normalized": normalized,
        "modifiers": modifiers,
        "vk": vk,
        "display": display,
        "error": "",
    }


def load_hotkey_from_env() -> tuple[int, int, str]:
    raw = os.environ.get("HOTKEY", DEFAULT_HOTKEY).strip()
    result = validate_hotkey_string(raw)
    if not result["ok"]:
        raise ValueError(f"Invalid HOTKEY in .env ({raw!r}): {result['error']}")
    return int(result["modifiers"]), int(result["vk"]), str(result["display"])
