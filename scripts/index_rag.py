"""Rebuild the local RAG index from context/ (run after adding or changing files)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from router.rag import rebuild_index  # noqa: E402

CONTEXT = ROOT / "context"
count = rebuild_index(CONTEXT, ROOT)
print(f"Indexed {count} chunks from {CONTEXT}")
