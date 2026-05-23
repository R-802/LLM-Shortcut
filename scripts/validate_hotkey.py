"""CLI helper for setup.ps1: validate HOTKEY string, print JSON."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from router.hotkey import validate_hotkey_string  # noqa: E402

raw = sys.argv[1] if len(sys.argv) > 1 else ""
print(json.dumps(validate_hotkey_string(raw)))
