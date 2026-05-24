"""Rebuild the local RAG index from context/ (run after adding or changing files)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Project-local embedding cache (before any chromadb import).
os.environ.setdefault(
    "CHROMA_CACHE_DIR", str((ROOT / ".chroma_cache").resolve())
)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from router.rag import consolidate_pending_index, rebuild_index  # noqa: E402

CONTEXT = ROOT / "context"
count = rebuild_index(CONTEXT, ROOT)
if consolidate_pending_index(ROOT):
    print("Consolidated rag_index_pending into rag_index/.")
print(f"Indexed {count} chunks from {CONTEXT}")
