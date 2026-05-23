"""Local RAG: chunk documents, embed with ChromaDB, retrieve relevant passages."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from router.documents import extract_text, iter_source_files

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_collection_name = "exam_context"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    return default


def is_enabled() -> bool:
    return _env_bool("RAG_ENABLED", default=True)


def top_k() -> int:
    try:
        return max(1, min(20, int(os.environ.get("RAG_TOP_K", "5"))))
    except ValueError:
        return 5


def chunk_size() -> int:
    try:
        return max(200, int(os.environ.get("RAG_CHUNK_CHARS", "700")))
    except ValueError:
        return 700


def chunk_overlap() -> int:
    try:
        return max(0, int(os.environ.get("RAG_CHUNK_OVERLAP", "120")))
    except ValueError:
        return 120


def index_dir(base: Path) -> Path:
    return base / "rag_index"


def manifest_path(base: Path) -> Path:
    return index_dir(base) / "manifest.json"


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _scan_signatures(source_dir: Path) -> dict[str, dict]:
    sigs: dict[str, dict] = {}
    for path in iter_source_files(source_dir):
        rel = path.relative_to(source_dir).as_posix()
        sigs[rel] = _file_signature(path)
    return sigs


def _load_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("files", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(path: Path, files: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"files": files}, indent=2),
        encoding="utf-8",
    )


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or chunk_size()
    overlap = overlap or chunk_overlap()
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at > start + size // 3:
                end = break_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _get_client(index_path: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(index_path),
        settings=Settings(anonymized_telemetry=False),
    )


def rebuild_index(source_dir: Path, base_dir: Path) -> int:
    """Re-ingest all files under source_dir. Returns chunk count."""
    import shutil

    source_dir = source_dir.resolve()
    idx_path = index_dir(base_dir)
    if idx_path.exists():
        shutil.rmtree(idx_path)
    idx_path.mkdir(parents=True, exist_ok=True)

    client = _get_client(idx_path)
    collection = client.get_or_create_collection(
        name=_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for file_path in iter_source_files(source_dir):
        rel = file_path.relative_to(source_dir).as_posix()
        text = extract_text(file_path).strip()
        if not text:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            chunk_id = f"{rel}::{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source": rel, "chunk": i})

    if documents:
        batch = 64
        for i in range(0, len(documents), batch):
            collection.add(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

    _save_manifest(manifest_path(base_dir), _scan_signatures(source_dir))
    logger.info("RAG index built: %d chunks from %s", len(documents), source_dir)
    return len(documents)


def ensure_index(source_dir: Path, base_dir: Path) -> None:
    """Rebuild index if source files changed."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        return

    current = _scan_signatures(source_dir)
    manifest_file = manifest_path(base_dir)
    previous = _load_manifest(manifest_file)

    if current == previous and (index_dir(base_dir) / "chroma.sqlite3").exists():
        return

    with _lock:
        current2 = _scan_signatures(source_dir)
        previous2 = _load_manifest(manifest_file)
        if current2 == previous2 and (index_dir(base_dir) / "chroma.sqlite3").exists():
            return
        rebuild_index(source_dir, base_dir)


def retrieve(question: str, source_dir: Path, base_dir: Path) -> str:
    """Return formatted retrieved chunks for the prompt."""
    question = (question or "").strip()
    if not question:
        return ""

    ensure_index(source_dir, base_dir)
    idx_path = index_dir(base_dir)
    if not (idx_path / "chroma.sqlite3").exists():
        logger.warning("RAG index missing; add files to %s and retry.", source_dir)
        return ""

    with _lock:
        client = _get_client(idx_path)
        try:
            collection = client.get_collection(_collection_name)
        except Exception:
            logger.warning("RAG collection not found.")
            return ""

        k = top_k()
        try:
            results = collection.query(query_texts=[question], n_results=k)
        except Exception as exc:
            logger.warning("RAG query failed: %s", exc)
            return ""

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    if not docs:
        return ""

    parts: list[str] = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        source = ""
        if metas and i < len(metas) and metas[i]:
            source = metas[i].get("source", "")
        header = f"[{source}]" if source else f"[chunk {i + 1}]"
        parts.append(f"{header}\n{doc.strip()}")

    logger.info("RAG retrieved %d chunk(s) for query.", len(parts))
    return "\n\n".join(parts)


def build_context_block(question: str, source_dir: Path, base_dir: Path) -> str:
    """Retrieved context for the user prompt."""
    if not is_enabled():
        return ""
    retrieved = retrieve(question, source_dir, base_dir)
    if not retrieved:
        return ""
    return f"Retrieved context (most relevant passages):\n{retrieved}"
