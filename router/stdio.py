"""Safe stdin/stdout/stderr when running under pythonw.exe (all are None)."""

from __future__ import annotations

import io
import sys


class DevNullIO(io.TextIOBase):
    """Discard writes; accept both str and bytes (logging/tracebacks vary)."""

    encoding = "utf-8"

    def write(self, s) -> int:  # type: ignore[override]
        if not s:
            return 0
        if isinstance(s, bytes):
            return len(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def patch_pythonw_stdio() -> None:
    if sys.stdin is None:
        sys.stdin = DevNullIO()
    if sys.stdout is None:
        sys.stdout = DevNullIO()
    if sys.stderr is None:
        sys.stderr = DevNullIO()
