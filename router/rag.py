"""Local RAG: chunk documents, embed with ChromaDB, retrieve relevant passages."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

from router.documents import extract_text, iter_source_files
from router.rag_sources import (
    expand_query_with_sources,
    infer_source_filters,
    prepare_chunk_for_embedding,
)
from router.stdio import patch_pythonw_stdio

logger = logging.getLogger(__name__)

_lock = None  # lazy threading.Lock()
_collection_name = "clip_context"
_client = None
_client_index_path: str | None = None

_rag_failures = 0
_rag_session_disabled = False
_MAX_RAG_FAILURES = 5


def _get_lock():
    global _lock
    if _lock is None:
        import threading
        _lock = threading.Lock()
    return _lock


def _query_timeout_s() -> float:
    try:
        return max(5.0, float(os.environ.get("RAG_QUERY_TIMEOUT_S", "30")))
    except ValueError:
        return 30.0


def _note_rag_failure(reason: str) -> None:
    global _rag_failures, _rag_session_disabled
    _rag_failures += 1
    if _rag_failures >= _MAX_RAG_FAILURES and not _rag_session_disabled:
        _rag_session_disabled = True
        logger.warning(
            "RAG disabled for this session after %d failures (last: %s). "
            "Restart Clip Assist or run scripts\\index_rag.bat.",
            _rag_failures,
            reason,
        )


def _note_rag_success() -> None:
    global _rag_failures, _rag_session_disabled
    _rag_failures = 0
    _rag_session_disabled = False


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


def _embedding_function(base_dir: Path):
    """
    Chroma's default ONNX embedder writes to ~/.cache/chroma, which often breaks under
    pythonw (permission errors). Force models into the project .chroma_cache folder.
    """
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    cache_root = (
        base_dir / ".chroma_cache" / "onnx_models" / "all-MiniLM-L6-v2"
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    ef = ONNXMiniLM_L6_V2()
    ef.DOWNLOAD_PATH = cache_root
    return ef


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


def _reset_index_dir(idx_path: Path) -> None:
    """Remove or rename index folder (handles Windows file locks from a running app)."""
    if not idx_path.exists():
        idx_path.mkdir(parents=True, exist_ok=True)
        return
    global _client, _client_index_path
    _client = None
    _client_index_path = None
    try:
        shutil.rmtree(idx_path)
    except OSError as exc:
        backup = idx_path.parent / f"rag_index_locked_{int(time.time())}"
        logger.warning("Could not delete %s (%s); moved to %s", idx_path, exc, backup.name)
        try:
            shutil.move(str(idx_path), str(backup))
        except OSError:
            raise RuntimeError(
                f"RAG index is locked. Stop Clip Assist (pythonw.exe), then run scripts\\index_rag.bat."
            ) from exc
    idx_path.mkdir(parents=True, exist_ok=True)


def _get_client(index_path: Path, base_dir: Path):
    global _client, _client_index_path
    patch_pythonw_stdio()

    path_str = str(index_path.resolve())
    if _client is not None and _client_index_path == path_str:
        return _client

    import chromadb
    from chromadb.config import Settings

    _client = chromadb.PersistentClient(
        path=path_str,
        settings=Settings(anonymized_telemetry=False),
    )
    _client_index_path = path_str
    return _client


def _get_collection(client, base_dir: Path, *, create: bool = False):
    ef = _embedding_function(base_dir)
    if create:
        return client.get_or_create_collection(
            name=_collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )
    return client.get_collection(
        name=_collection_name,
        embedding_function=ef,
    )


def rebuild_index(source_dir: Path, base_dir: Path) -> int:
    """Re-ingest all files under source_dir. Returns chunk count."""
    source_dir = source_dir.resolve()
    base_dir = base_dir.resolve()
    idx_path = index_dir(base_dir)

    _reset_index_dir(idx_path)

    client = _get_client(idx_path, base_dir)
    collection = _get_collection(client, base_dir, create=True)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for file_path in iter_source_files(source_dir):
        rel = file_path.relative_to(source_dir).as_posix()
        text = extract_text(file_path).strip()
        if not text:
            continue
        doc_title = Path(rel).name
        for i, chunk in enumerate(chunk_text(text)):
            chunk_id = f"{rel}::{i}"
            ids.append(chunk_id)
            documents.append(prepare_chunk_for_embedding(rel, chunk))
            metadatas.append({
                "source": rel,
                "chunk": i,
                "filename": doc_title,
            })

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
    """Rebuild index if source files changed (used by index_rag.py / setup, not hotkey path)."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        return

    current = _scan_signatures(source_dir)
    manifest_file = manifest_path(base_dir)
    previous = _load_manifest(manifest_file)

    if current == previous and (index_dir(base_dir) / "chroma.sqlite3").exists():
        return

    with _get_lock():
        current2 = _scan_signatures(source_dir)
        previous2 = _load_manifest(manifest_file)
        if current2 == previous2 and (index_dir(base_dir) / "chroma.sqlite3").exists():
            return
        rebuild_index(source_dir, base_dir)


def _query_collection(
    collection,
    question: str,
    k: int,
    *,
    source_filter: list[str] | None = None,
) -> dict:
    """Run query on the calling thread (Chroma sqlite is not safe across threads)."""
    kwargs: dict = {"query_texts": [question], "n_results": k}
    if source_filter:
        if len(source_filter) == 1:
            kwargs["where"] = {"source": {"$eq": source_filter[0]}}
        else:
            kwargs["where"] = {"source": {"$in": source_filter}}
    return collection.query(**kwargs)


def _merge_query_results(primary: dict, extra: dict, k: int) -> tuple[list, list]:
    """Combine document lists, dedupe by id, keep order (primary first)."""
    ids_p = primary.get("ids") or [[]]
    docs_p = primary.get("documents") or [[]]
    metas_p = primary.get("metadatas") or [[]]
    if not ids_p[0]:
        return docs_p[0], metas_p[0]

    seen = set(ids_p[0])
    docs = list(docs_p[0])
    metas = list(metas_p[0])

    for doc_id, doc, meta in zip(
        (extra.get("ids") or [[]])[0],
        (extra.get("documents") or [[]])[0],
        (extra.get("metadatas") or [[]])[0],
    ):
        if len(docs) >= k:
            break
        if doc_id in seen or not doc:
            continue
        seen.add(doc_id)
        docs.append(doc)
        metas.append(meta)
    return docs, metas


def retrieve(question: str, source_dir: Path, base_dir: Path) -> str:
    """Return formatted retrieved chunks for the prompt."""
    global _rag_session_disabled

    question = (question or "").strip()
    if not question or _rag_session_disabled:
        return ""

    source_dir = source_dir.resolve()
    base_dir = base_dir.resolve()

    idx_path = index_dir(base_dir)
    if not (idx_path / "chroma.sqlite3").exists():
        logger.warning(
            "RAG index missing; run scripts\\index_rag.bat after adding files to %s",
            source_dir,
        )
        return ""

    t0 = time.perf_counter()
    try:
        with _get_lock():
            client = _get_client(idx_path, base_dir)
            try:
                collection = _get_collection(client, base_dir)
            except Exception:
                logger.warning("RAG collection not found; run scripts\\index_rag.bat.")
                return ""

            count = collection.count()
            if count == 0:
                logger.warning(
                    "RAG index is empty (0 chunks). Run scripts\\index_rag.bat."
                )
                return ""

            k = min(top_k(), count)
            known_sources = [
                p.relative_to(source_dir.resolve()).as_posix()
                for p in iter_source_files(source_dir)
            ]
            source_filter = infer_source_filters(question, known_sources)
            query_text = expand_query_with_sources(question, source_filter)

            if source_filter:
                logger.info(
                    "RAG document filter: %s",
                    ", ".join(source_filter),
                )
                results = _query_collection(
                    collection, query_text, k, source_filter=source_filter
                )
                docs_f = (results.get("documents") or [[]])[0]
                if not docs_f:
                    # Old index or few chunks — pull directly from matched file(s)
                    pull_docs: list = []
                    pull_metas: list = []
                    for src in source_filter:
                        got = collection.get(
                            where={"source": {"$eq": src}},
                            include=["documents", "metadatas"],
                        )
                        pull_docs.extend(got.get("documents") or [])
                        pull_metas.extend(got.get("metadatas") or [])
                    if pull_docs:
                        results = {
                            "documents": [pull_docs[:k]],
                            "metadatas": [pull_metas[:k]],
                        }
                elif len(docs_f) < k:
                    extra = _query_collection(
                        collection,
                        query_text,
                        k * 2,
                        source_filter=source_filter,
                    )
                    docs, metas = _merge_query_results(results, extra, k)
                    results = {
                        "documents": [docs],
                        "metadatas": [metas],
                    }
            else:
                results = _query_collection(collection, query_text, k)
    except Exception as exc:
        logger.warning("RAG query failed: %s", exc)
        _note_rag_failure(str(exc))
        return ""

    elapsed = time.perf_counter() - t0
    if elapsed > _query_timeout_s():
        logger.warning(
            "RAG query slow (%.1fs > %.0fs limit) but completed.",
            elapsed,
            _query_timeout_s(),
        )

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

    sources_used = list(dict.fromkeys(
        m.get("source", "") for m in (metas or []) if m and m.get("source")
    ))
    logger.info(
        "RAG retrieved %d chunk(s) in %.2fs from [%s] (index %d chunks).",
        len(parts),
        elapsed,
        ", ".join(sources_used) or "unknown",
        count,
    )
    _note_rag_success()
    return "\n\n".join(parts)


def warmup(base_dir: Path, source_dir: Path) -> None:
    """Load embedding model and Chroma client (call once at app startup)."""
    if not is_enabled() or not source_dir.is_dir():
        return
    idx_path = index_dir(base_dir)
    if not (idx_path / "chroma.sqlite3").exists():
        logger.info("RAG warmup skipped: no index (run scripts\\index_rag.bat).")
        return
    try:
        with _get_lock():
            client = _get_client(idx_path, base_dir)
            collection = _get_collection(client, base_dir)
            n = collection.count()
        if n > 0:
            retrieve("warmup query for embedding model", source_dir, base_dir)
            logger.info("RAG warmup complete (%d chunks indexed).", n)
    except Exception as exc:
        logger.warning("RAG warmup failed: %s", exc)


def build_context_block(question: str, source_dir: Path, base_dir: Path) -> str:
    """Retrieved context for the user prompt."""
    if not is_enabled():
        return ""
    retrieved = retrieve(question, source_dir, base_dir)
    if not retrieved:
        return ""
    return f"Retrieved context (most relevant passages):\n{retrieved}"
