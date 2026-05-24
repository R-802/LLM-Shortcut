"""Build a Chroma index in a child process (Windows releases file locks on exit)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: python -m router.rag_populate INDEX_DIR SOURCE_DIR BASE_DIR",
            file=sys.stderr,
        )
        return 2

    idx_path = Path(sys.argv[1]).resolve()
    source_dir = Path(sys.argv[2]).resolve()
    base_dir = Path(sys.argv[3]).resolve()

    root = base_dir
    if (root / ".env").is_file():
        from dotenv import load_dotenv

        load_dotenv(root / ".env")

    import os

    os.environ.setdefault(
        "CHROMA_CACHE_DIR", str((root / ".chroma_cache").resolve())
    )

    from router.rag import populate_index_inprocess

    count = populate_index_inprocess(idx_path, source_dir, base_dir)
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
