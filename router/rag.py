"""Local RAG: chunk documents, embed with ChromaDB, retrieve relevant passages."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from router.documents import extract_text, iter_source_files
from router.embeddings import embedding_fingerprint, get_embedding_function
from router.rag_sources import (
    chunk_body_for_prompt,
    expand_query_with_sources,
    find_companion_solutions,
    infer_source_filters,
    match_explicit_filenames,
    prepare_chunk_for_embedding,
    slice_xlsx_for_question,
)
from router.stdio import patch_pythonw_stdio

logger = logging.getLogger(__name__)

_lock = None  # lazy threading.Lock() — Chroma queries + index swap only
_rebuild_schedule_lock = None  # prevents parallel full rebuilds
_collection_name = "clip_context"
_build_lock_name = ".rag_build.lock"
_client = None
_client_index_path: str | None = None

_rag_failures = 0
_rag_session_disabled = False
_MAX_RAG_FAILURES = 5
_warmup_done = None  # lazy threading.Event()


def _get_lock():
    global _lock
    if _lock is None:
        import threading
        _lock = threading.Lock()
    return _lock


def _get_rebuild_schedule_lock():
    global _rebuild_schedule_lock
    if _rebuild_schedule_lock is None:
        import threading
        _rebuild_schedule_lock = threading.Lock()
    return _rebuild_schedule_lock


def _lock_timeout_s() -> float:
    try:
        return max(0.5, float(os.environ.get("RAG_LOCK_TIMEOUT_S", "3")))
    except ValueError:
        return 3.0


def _get_warmup_done():
    global _warmup_done
    if _warmup_done is None:
        import threading
        _warmup_done = threading.Event()
    return _warmup_done


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


def max_distance() -> float:
    """
    Max cosine distance (Chroma hnsw:space=cosine) for a chunk to be included.
    Lower = stricter. Typical: good match ~0.5-0.7, weak ~0.85+.
    """
    try:
        return float(os.environ.get("RAG_MAX_DISTANCE", "0.82"))
    except ValueError:
        return 0.82


def max_distance_open() -> float:
    """Stricter limit when the question does not name a specific context/ file."""
    try:
        return float(os.environ.get("RAG_MAX_DISTANCE_OPEN", "0.62"))
    except ValueError:
        return 0.62


def open_query_min_term_hits() -> int:
    """Distinctive keywords a chunk must share for an open (unnamed) query."""
    try:
        return max(1, int(os.environ.get("RAG_OPEN_TERM_MIN_HITS", "2")))
    except ValueError:
        return 2


_RAG_TERM_STOPWORDS = frozenset({
    "about", "accounts", "actual", "after", "also", "application", "applications",
    "approximately", "assume", "assuming", "because", "being", "benefit", "between",
    "bought", "buying", "calculated", "capacity", "change", "coefficient", "compare",
    "conditions", "consume", "context", "conventional", "cost", "define", "delivered",
    "delivers", "density", "describe", "difference", "discuss", "efficiency",
    "efficiencies", "electricity", "energy", "engine", "engines", "everything",
    "explain", "factor", "files", "fluid", "following", "generation", "generator",
    "heat", "heater", "heating", "house", "households", "important", "installed",
    "many", "marks", "model", "much", "normally", "often", "operates", "operating",
    "other", "overall", "performance", "plant", "plants", "power", "primary", "pump",
    "question", "receive", "report", "reservoir", "residential", "results", "should",
    "show", "station", "stations", "stove", "such", "summarize", "summary", "system",
    "systems", "table", "technology", "technologies", "tell", "temperature", "that",
    "their", "there", "thermal", "these", "they", "this", "turbine", "under", "used",
    "using", "value", "volume", "what", "when", "where", "which", "while", "with",
    "would", "your", "zealand",
})


def _distinct_question_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for token in re.split(r"\W+", question.lower()):
        if len(token) >= 4 and token not in _RAG_TERM_STOPWORDS and not token.isdigit():
            terms.add(token)
    return terms


def _term_in_text(term: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(term)}", text, re.IGNORECASE) is not None


def _filter_chunks_by_question_terms(
    question: str,
    docs: list,
    metas: list,
    *,
    min_hits: int = 1,
) -> tuple[list, list]:
    """Drop chunks that share no distinctive keywords with the question."""
    terms = _distinct_question_terms(question)
    if not terms:
        return docs, metas

    specific = {t for t in terms if len(t) >= 8}

    kept_docs: list = []
    kept_metas: list = []
    for doc, meta in zip(docs, metas):
        body = chunk_body_for_prompt(str(doc)).lower()
        hits = sum(1 for term in terms if _term_in_text(term, body))
        if hits < min_hits:
            continue
        if specific and not any(_term_in_text(term, body) for term in specific):
            continue
        kept_docs.append(doc)
        kept_metas.append(meta)
    return kept_docs, kept_metas


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


_LIVE_INDEX_NAME = "rag_index"
_PENDING_INDEX_NAME = "rag_index_pending"
_ACTIVE_INDEX_MARKER = ".rag_index_active"


def live_index_dir(base: Path) -> Path:
    return base.resolve() / _LIVE_INDEX_NAME


def index_dir(base: Path) -> Path:
    """Active Chroma directory (live, pending override, or whichever has a DB)."""
    base = base.resolve()
    marker = base / _ACTIVE_INDEX_MARKER
    if marker.is_file():
        rel = marker.read_text(encoding="utf-8").strip()
        if rel:
            path = base / rel
            if (path / "chroma.sqlite3").is_file():
                return path
    live = live_index_dir(base)
    if (live / "chroma.sqlite3").is_file():
        return live
    pending = base / _PENDING_INDEX_NAME
    if (pending / "chroma.sqlite3").is_file():
        return pending
    return live


def manifest_path(base: Path) -> Path:
    return index_dir(base) / "manifest.json"


def _set_active_index(base: Path, folder_name: str) -> None:
    (base / _ACTIVE_INDEX_MARKER).write_text(folder_name, encoding="utf-8")


def _clear_active_index(base: Path) -> None:
    try:
        (base / _ACTIVE_INDEX_MARKER).unlink(missing_ok=True)
    except OSError:
        pass


def consolidate_pending_index(base_dir: Path) -> bool:
    """
    Move rag_index_pending/ over rag_index/ when the live folder is not locked.
    Returns True when consolidated.
    """
    base_dir = base_dir.resolve()
    pending = base_dir / _PENDING_INDEX_NAME
    live = live_index_dir(base_dir)
    if not (pending / "chroma.sqlite3").is_file():
        return False

    _release_filesystem_locks()
    backup = base_dir / f"rag_index_prev_{int(time.time())}"
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)

    try:
        if live.exists():
            shutil.move(str(live), str(backup))
        shutil.move(str(pending), str(live))
        _clear_active_index(base_dir)
        logger.info("RAG: consolidated %s into %s/.", _PENDING_INDEX_NAME, _LIVE_INDEX_NAME)
        return True
    except OSError as exc:
        logger.debug("RAG pending consolidate skipped: %s", exc)
        return False


def _embedding_function(base_dir: Path):
    """Chroma Cloud embeddings when CHROMA_API_KEY is set; else local ONNX MiniLM."""
    return get_embedding_function(base_dir)


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _scan_signatures(source_dir: Path) -> dict[str, dict]:
    sigs: dict[str, dict] = {}
    for path in iter_source_files(source_dir):
        rel = path.relative_to(source_dir).as_posix()
        sigs[rel] = _file_signature(path)
    return sigs


def _read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_manifest(path: Path) -> dict[str, dict]:
    return _read_manifest(path).get("files", {})


def _save_manifest(path: Path, files: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"files": files, "embedding": embedding_fingerprint()},
            indent=2,
        ),
        encoding="utf-8",
    )


def _prompt_max_chars() -> int:
    try:
        return max(2000, int(os.environ.get("RAG_PROMPT_MAX_CHARS", "10000")))
    except ValueError:
        return 10000


def _merge_chunk_bodies(bodies: list[str]) -> str:
    """Join passages from the same file; drop overlap duplicates between chunks."""
    if not bodies:
        return ""
    parts = [bodies[0].strip()]
    for body in bodies[1:]:
        body = body.strip()
        if not body:
            continue
        prev = parts[-1]
        if body in prev:
            continue
        appended = False
        max_ov = min(len(prev), len(body), 400)
        for n in range(max_ov, 40, -1):
            if prev[-n:] == body[:n]:
                tail = body[n:].strip()
                if tail:
                    parts.append(tail)
                appended = True
                break
        if not appended:
            parts.append(body)
    return "\n\n".join(p for p in parts if p)


def format_retrieved_passages(docs: list, metas: list) -> str:
    """Group chunks by file, strip index metadata, cap total length."""
    by_source: dict[str, list[str]] = {}
    for doc, meta in zip(docs, metas):
        if not doc:
            continue
        source = ""
        if meta and isinstance(meta, dict):
            source = meta.get("source", "") or ""
        body = chunk_body_for_prompt(str(doc))
        if not body:
            continue
        by_source.setdefault(source or "unknown", []).append(body)

    sections: list[str] = []
    for source, bodies in by_source.items():
        label = Path(source).name if source and source != "unknown" else source
        merged = _merge_chunk_bodies(bodies)
        if merged:
            sections.append(f"--- {label} ---\n{merged}")

    text = "\n\n".join(sections)
    max_chars = _prompt_max_chars()
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut].rstrip() + "\n\n[... context truncated ...]"


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or chunk_size()
    overlap = overlap or chunk_overlap()
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at <= start + size // 3:
                break_at = text.rfind(". ", start, end)
                if break_at > start + size // 3:
                    break_at += 1
            if break_at > start + size // 3:
                end = break_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _close_client() -> None:
    """Release Chroma handles so rag_index/ can be rebuilt in-process on Windows."""
    global _client, _client_index_path
    if _client is not None:
        try:
            reset = getattr(_client, "reset", None)
            if callable(reset):
                reset()
        except Exception:
            pass
    _client = None
    _client_index_path = None
    gc.collect()


def _reset_index_dir(idx_path: Path) -> None:
    """Remove or rename index folder (handles Windows file locks from a running app)."""
    if not idx_path.exists():
        idx_path.mkdir(parents=True, exist_ok=True)
        return
    _close_client()
    last_exc: OSError | None = None
    for attempt in range(4):
        try:
            if idx_path.exists():
                shutil.rmtree(idx_path)
            idx_path.mkdir(parents=True, exist_ok=True)
            return
        except OSError as exc:
            last_exc = exc
            gc.collect()
            time.sleep(0.25 * (attempt + 1))

    backup = idx_path.parent / f"rag_index_locked_{int(time.time())}"
    try:
        shutil.move(str(idx_path), str(backup))
        logger.warning(
            "Could not delete %s (%s); moved aside to %s",
            idx_path,
            last_exc,
            backup.name,
        )
    except OSError as move_exc:
        raise RuntimeError(
            "RAG index is locked. Stop Clip Assist (pythonw.exe), then run scripts\\index_rag.bat."
        ) from move_exc
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


def _named_file_max_chars() -> int:
    try:
        return max(2000, int(os.environ.get("RAG_NAMED_FILE_MAX_CHARS", "12000")))
    except ValueError:
        return 12000


def populate_index_inprocess(
    idx_path: Path, source_dir: Path, base_dir: Path
) -> int:
    """Write chunks into a fresh index directory (in-process; used by child on Windows)."""
    import chromadb
    from chromadb.config import Settings

    idx_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(idx_path.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )
    ef = _embedding_function(base_dir)
    collection = client.get_or_create_collection(
        name=_collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )

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

    count = len(documents)
    del collection
    del client
    gc.collect()
    return count


def _populate_index_at(idx_path: Path, source_dir: Path, base_dir: Path) -> int:
    """
    Build index at idx_path.

    On Windows, run in a subprocess so SQLite/HNSW files are not locked during promote.
    """
    if os.name != "nt":
        return populate_index_inprocess(idx_path, source_dir, base_dir)

    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "router.rag_populate",
        str(idx_path.resolve()),
        str(source_dir.resolve()),
        str(base_dir.resolve()),
    ]
    env = os.environ.copy()
    env.setdefault(
        "CHROMA_CACHE_DIR",
        str((base_dir / ".chroma_cache").resolve()),
    )
    result = subprocess.run(
        cmd,
        cwd=str(base_dir.resolve()),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"RAG index subprocess failed (exit {result.returncode}): {detail}"
        )
    return int((result.stdout or "0").strip().splitlines()[-1])


def _release_filesystem_locks() -> None:
    _close_client()
    for _ in range(3):
        gc.collect()
        time.sleep(0.35)


def _promote_staging_index(staging: Path, idx_path: Path, base_dir: Path) -> None:
    """
    Install staging index: copy to rag_index/ when possible, else rag_index_pending/
    with an active-index marker (Clip Assist can query pending while live is locked).
    """
    _release_filesystem_locks()
    pending = base_dir / _PENDING_INDEX_NAME
    if pending.exists():
        shutil.rmtree(pending, ignore_errors=True)

    last_exc: OSError | None = None
    for attempt in range(6):
        try:
            shutil.copytree(staging, pending)
            last_exc = None
            break
        except OSError as exc:
            last_exc = exc
            _release_filesystem_locks()
            time.sleep(0.5 * (attempt + 1))
    if last_exc is not None:
        raise RuntimeError(
            "Could not copy RAG staging index. Stop Clip Assist and retry."
        ) from last_exc

    live = idx_path
    installed_live = False
    if live.exists():
        backup = base_dir / f"rag_index_prev_{int(time.time())}"
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        for attempt in range(6):
            try:
                shutil.move(str(live), str(backup))
                shutil.move(str(pending), str(live))
                _clear_active_index(base_dir)
                installed_live = True
                break
            except OSError:
                _release_filesystem_locks()
                time.sleep(0.5 * (attempt + 1))
    else:
        try:
            shutil.move(str(pending), str(live))
            _clear_active_index(base_dir)
            installed_live = True
        except OSError:
            pass

    if not installed_live:
        _set_active_index(base_dir, _PENDING_INDEX_NAME)
        logger.warning(
            "RAG: %s is locked; installed new index at %s/. "
            "Stop Clip Assist and run scripts\\index_rag.bat to consolidate.",
            _LIVE_INDEX_NAME,
            _PENDING_INDEX_NAME,
        )

    for attempt in range(6):
        try:
            _clear_staging_dir(staging)
            break
        except OSError:
            _release_filesystem_locks()
            time.sleep(0.5 * (attempt + 1))
    else:
        logger.warning("RAG staging folder could not be removed; safe to delete %s", staging)


def _try_named_disk_context(
    source_dir: Path,
    base_dir: Path,
    source_filter: list[str],
    *,
    reason: str,
    question: str = "",
    force: bool = False,
) -> tuple[str, str, list[str]] | None:
    """Load named context/ files without opening Chroma (fast path during rebuild)."""
    if not source_filter:
        return None

    if force:
        load_list = list(source_filter)
    else:
        manifest_files = _load_manifest(manifest_path(base_dir))
        not_indexed = [s for s in source_filter if s not in manifest_files]
        stale = index_is_stale(source_dir, base_dir)
        load_list = source_filter if stale else not_indexed
        if not load_list:
            return None

    disk_text, loaded = _load_named_sources_from_disk(
        source_dir, load_list, question=question
    )
    if not disk_text:
        return None
    logger.info("RAG loaded from disk (%s): %s", reason, ", ".join(loaded))
    return disk_text, "ok", loaded


def _load_named_sources_from_disk(
    source_dir: Path,
    sources: list[str],
    *,
    question: str = "",
) -> tuple[str, list[str]]:
    """Read named context/ files when they are not in the Chroma index yet."""
    max_chars = _named_file_max_chars()
    parts: list[str] = []
    loaded: list[str] = []
    for rel in sources:
        path = source_dir / rel
        if not path.is_file():
            logger.warning("RAG: named file not found on disk: %s", rel)
            continue
        text = extract_text(path).strip()
        if path.suffix.lower() in (".xlsx", ".xls") and question:
            text = slice_xlsx_for_question(text, question)
        if not text:
            logger.warning("RAG: named file empty or unreadable: %s", rel)
            continue
        label = Path(rel).name
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n\n[... truncated ...]"
        parts.append(f"--- {label} ---\n{text}")
        loaded.append(rel)
    combined = "\n\n".join(parts)
    cap = _prompt_max_chars()
    if len(combined) > cap:
        combined = combined[:cap].rstrip() + "\n\n[... context truncated ...]"
    return combined, loaded


def _clear_staging_dir(staging: Path) -> None:
    if not staging.exists():
        return
    for attempt in range(5):
        try:
            shutil.rmtree(staging)
            return
        except OSError:
            gc.collect()
            time.sleep(0.3 * (attempt + 1))
    raise OSError(f"Could not clear RAG staging folder: {staging}")


def _staging_build_in_progress(staging: Path) -> bool:
    """True only while an active rebuild holds .rag_build.lock (not merely partial files)."""
    started = _staging_build_started(staging)
    if started is None:
        return False
    return (time.time() - started) <= _build_timeout_s()


def _recover_stale_staging(staging: Path) -> None:
    if not staging.exists():
        return
    if _staging_build_in_progress(staging):
        return
    if _staging_build_started(staging) is not None or (staging / "chroma.sqlite3").is_file():
        logger.warning(
            "RAG clearing abandoned staging folder (build timed out after %.0fs).",
            _build_timeout_s(),
        )
        _clear_staging_dir(staging)


def rebuild_index(source_dir: Path, base_dir: Path) -> int | None:
    """Re-ingest all files under source_dir. Returns chunk count, or None if skipped."""
    source_dir = source_dir.resolve()
    base_dir = base_dir.resolve()
    idx_path = live_index_dir(base_dir)
    staging = base_dir / "rag_index_staging"

    _recover_stale_staging(staging)

    if _staging_build_in_progress(staging):
        logger.info("RAG rebuild skipped: staging index build already in progress.")
        return None

    nested_staging = idx_path / "rag_index_staging"
    if nested_staging.exists():
        shutil.rmtree(nested_staging, ignore_errors=True)

    _clear_staging_dir(staging)
    staging.mkdir(parents=True)
    _staging_build_lock(staging).write_text(str(time.time()), encoding="utf-8")

    try:
        count = _populate_index_at(staging, source_dir, base_dir)
        _save_manifest(staging / "manifest.json", _scan_signatures(source_dir))

        with _get_lock():
            _close_client()
            _promote_staging_index(staging, idx_path, base_dir)

        _save_manifest(manifest_path(base_dir), _scan_signatures(source_dir))
        logger.info("RAG index built: %d chunks from %s", count, source_dir)
        return count
    finally:
        lock = _staging_build_lock(staging)
        if lock.is_file():
            try:
                lock.unlink()
            except OSError:
                pass


def index_is_stale(source_dir: Path, base_dir: Path) -> bool:
    """True when context/ files differ from the last indexed manifest."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        return False
    manifest = _read_manifest(manifest_path(base_dir))
    if manifest.get("embedding") != embedding_fingerprint():
        logger.info(
            "RAG index stale: embedding model changed (%s -> %s).",
            manifest.get("embedding", "none"),
            embedding_fingerprint(),
        )
        return True
    current = _scan_signatures(source_dir)
    previous = manifest.get("files", {})
    if current != previous:
        return True
    return not (index_dir(base_dir) / "chroma.sqlite3").exists()


def ensure_index(source_dir: Path, base_dir: Path) -> int | None:
    """
    Rebuild index if source files changed.

    Returns chunk count when a rebuild ran, else None.
    Embedding runs outside the Chroma lock so hotkeys stay responsive.
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        return None

    if not index_is_stale(source_dir, base_dir):
        return None

    staging = base_dir / "rag_index_staging"
    _recover_stale_staging(staging)
    if _staging_build_in_progress(staging):
        logger.info("RAG rebuild skipped: staging index build already in progress.")
        return None

    schedule = _get_rebuild_schedule_lock()
    if not schedule.acquire(blocking=False):
        logger.info("RAG rebuild already in progress; skipping duplicate.")
        return None
    try:
        if not index_is_stale(source_dir, base_dir):
            return None
        return rebuild_index(source_dir, base_dir)
    finally:
        schedule.release()


def _watch_interval_s() -> float:
    try:
        return max(5.0, float(os.environ.get("RAG_WATCH_INTERVAL_S", "15")))
    except ValueError:
        return 15.0


def _watch_debounce_s() -> float:
    try:
        return max(1.0, float(os.environ.get("RAG_WATCH_DEBOUNCE_S", "3")))
    except ValueError:
        return 3.0


def _build_timeout_s() -> float:
    try:
        return max(120.0, float(os.environ.get("RAG_BUILD_TIMEOUT_S", "900")))
    except ValueError:
        return 900.0


def _staging_build_lock(staging: Path) -> Path:
    return staging / _build_lock_name


def _staging_build_started(staging: Path) -> float | None:
    lock = _staging_build_lock(staging)
    if not lock.is_file():
        return None
    try:
        return float(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _staging_build_timed_out(staging: Path) -> bool:
    started = _staging_build_started(staging)
    if started is not None:
        return (time.time() - started) > _build_timeout_s()
    sqlite = staging / "chroma.sqlite3"
    if sqlite.is_file():
        return (time.time() - sqlite.stat().st_mtime) > _build_timeout_s()
    return False


def _index_embedding_matches(base_dir: Path) -> bool:
    manifest = _read_manifest(manifest_path(base_dir))
    return manifest.get("embedding") == embedding_fingerprint()


def _watch_enabled() -> bool:
    return _env_bool("RAG_WATCH_ENABLED", default=True)


def start_index_watcher(source_dir: Path, base_dir: Path):
    """
    Background poll of context/; rebuilds the RAG index when files change.

    Returns the daemon thread, or None if watching is disabled.
    """
    if not is_enabled() or not _watch_enabled():
        return None

    import threading

    def _loop() -> None:
        if not _get_warmup_done().wait(timeout=180):
            logger.warning("RAG index watcher: warmup did not finish in 180s.")
        pending_sigs: dict[str, dict] | None = None
        pending_since = 0.0
        debounce = _watch_debounce_s()
        interval = _watch_interval_s()

        while True:
            time.sleep(interval)
            try:
                if not source_dir.is_dir():
                    pending_sigs = None
                    continue

                current = _scan_signatures(source_dir)
                if not index_is_stale(source_dir, base_dir):
                    pending_sigs = None
                    continue

                if pending_sigs != current:
                    pending_sigs = current
                    pending_since = time.time()
                    logger.info(
                        "RAG: context/ changed (%d file(s)); re-index in %.0fs if stable.",
                        len(current),
                        debounce,
                    )
                    continue

                if time.time() - pending_since < debounce:
                    continue

                count = ensure_index(source_dir, base_dir)
                pending_sigs = None
                if count is not None:
                    logger.info(
                        "RAG: auto re-indexed %d chunk(s) from context/.", count
                    )
            except Exception as exc:
                logger.warning("RAG index watcher error: %s", exc)
                pending_sigs = None

    thread = threading.Thread(
        target=_loop,
        daemon=True,
        name="rag-index-watcher",
    )
    thread.start()
    return thread


def _query_collection(
    collection,
    question: str,
    k: int,
    *,
    source_filter: list[str] | None = None,
) -> dict:
    """Run query on the calling thread (Chroma sqlite is not safe across threads)."""
    kwargs: dict = {
        "query_texts": [question],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if source_filter:
        if len(source_filter) == 1:
            kwargs["where"] = {"source": {"$eq": source_filter[0]}}
        else:
            kwargs["where"] = {"source": {"$in": source_filter}}
    return collection.query(**kwargs)


def _filter_results_by_distance(
    results: dict, limit: float, *, named_document: bool = False
) -> dict:
    """Drop chunks whose cosine distance exceeds the relevance threshold."""
    docs = list((results.get("documents") or [[]])[0])
    metas = list((results.get("metadatas") or [[]])[0])
    dists = list((results.get("distances") or [[]])[0])
    ids = list((results.get("ids") or [[]])[0])

    if not docs or not dists:
        return results

    best = min(dists)
    if best > limit:
        if named_document:
            logger.info(
                "RAG below distance threshold for named document "
                "(best %.3f > max %.3f); loading full file instead.",
                best,
                limit,
            )
        else:
            logger.info(
                "RAG below relevance threshold: best distance %.3f > max %.3f "
                "(question likely unrelated to context/).",
                best,
                limit,
            )
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

    kept_docs: list = []
    kept_metas: list = []
    kept_dists: list = []
    kept_ids: list = []
    for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
        if dist > limit or not doc:
            continue
        kept_docs.append(doc)
        kept_metas.append(meta)
        kept_dists.append(dist)
        kept_ids.append(doc_id)

    if len(kept_docs) < len(docs):
        logger.info(
            "RAG kept %d/%d chunk(s) within distance %.3f (best=%.3f).",
            len(kept_docs),
            len(docs),
            limit,
            best,
        )

    return {
        "documents": [kept_docs],
        "metadatas": [kept_metas],
        "distances": [kept_dists],
        "ids": [kept_ids],
    }


def _merge_query_results(primary: dict, extra: dict, k: int, limit: float) -> dict:
    """Combine result lists, dedupe by id, apply distance threshold."""
    ids_p = (primary.get("ids") or [[]])[0]
    if not ids_p:
        return extra

    seen = set(ids_p)
    docs = list((primary.get("documents") or [[]])[0])
    metas = list((primary.get("metadatas") or [[]])[0])
    dists = list((primary.get("distances") or [[]])[0])
    ids = list(ids_p)

    for doc_id, doc, meta, dist in zip(
        (extra.get("ids") or [[]])[0],
        (extra.get("documents") or [[]])[0],
        (extra.get("metadatas") or [[]])[0],
        (extra.get("distances") or [[]])[0],
    ):
        if len(docs) >= k:
            break
        if doc_id in seen or not doc or dist > limit:
            continue
        seen.add(doc_id)
        docs.append(doc)
        metas.append(meta)
        dists.append(dist)
        ids.append(doc_id)

    merged = {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists],
        "ids": [ids],
    }
    return _filter_results_by_distance(merged, limit)


def retrieve(question: str, source_dir: Path, base_dir: Path) -> tuple[str, str, list[str]]:
    """
    Return (formatted_chunks, status, source_paths).

    status is one of: ok, below_threshold, empty_index, error, disabled, busy.
    """
    global _rag_session_disabled

    question = (question or "").strip()
    if not question or _rag_session_disabled:
        return "", "disabled", []

    source_dir = source_dir.resolve()
    base_dir = base_dir.resolve()

    known_sources = [
        p.relative_to(source_dir).as_posix()
        for p in iter_source_files(source_dir)
    ]
    source_filter = infer_source_filters(question, known_sources)
    explicit_files = match_explicit_filenames(question, known_sources)

    if explicit_files:
        disk_hit = _try_named_disk_context(
            source_dir,
            base_dir,
            explicit_files,
            reason="file named in question",
            question=question,
            force=True,
        )
        if disk_hit:
            return disk_hit
        source_filter = explicit_files

    disk_hit = _try_named_disk_context(
        source_dir,
        base_dir,
        source_filter,
        reason="index pending or not indexed",
        question=question,
    )
    if disk_hit:
        return disk_hit

    idx_path = index_dir(base_dir)
    if not (idx_path / "chroma.sqlite3").exists():
        disk_hit = _try_named_disk_context(
            source_dir, base_dir, source_filter, reason="no index yet"
        )
        if disk_hit:
            return disk_hit
        logger.warning(
            "RAG index missing; run scripts\\index_rag.bat after adding files to %s",
            source_dir,
        )
        return "", "empty_index", []

    if not _index_embedding_matches(base_dir):
        disk_hit = _try_named_disk_context(
            source_dir,
            base_dir,
            source_filter,
            reason="embedding migration (re-index in progress)",
        )
        if disk_hit:
            return disk_hit
        logger.info(
            "RAG index uses %s; current config is %s — skipping vector search until re-index completes.",
            _read_manifest(manifest_path(base_dir)).get("embedding", "none"),
            embedding_fingerprint(),
        )
        return "", "busy", []

    t0 = time.perf_counter()
    lock = _get_lock()
    if not lock.acquire(timeout=_lock_timeout_s()):
        disk_hit = _try_named_disk_context(
            source_dir, base_dir, source_filter, reason="index busy"
        )
        if disk_hit:
            return disk_hit
        logger.warning(
            "RAG index busy (rebuild in progress); no context attached."
        )
        return "", "busy", []

    try:
        client = _get_client(idx_path, base_dir)
        try:
            collection = _get_collection(client, base_dir)
        except Exception:
            logger.warning("RAG collection not found; run scripts\\index_rag.bat.")
            return "", "error", []

        count = collection.count()
        if count == 0:
            logger.warning(
                "RAG index is empty (0 chunks). Run scripts\\index_rag.bat."
            )
            return "", "empty_index", []

        k = min(top_k(), count)
        query_text = expand_query_with_sources(question, source_filter)
        dist_limit = max_distance() if source_filter else max_distance_open()

        if source_filter:
            logger.info(
                "RAG document filter: %s",
                ", ".join(source_filter),
            )
            results = _filter_results_by_distance(
                _query_collection(
                    collection, query_text, k, source_filter=source_filter
                ),
                dist_limit,
                named_document=True,
            )
            docs_f = (results.get("documents") or [[]])[0]
            if not docs_f:
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
                    logger.info(
                        "RAG using %d chunk(s) from named document(s) "
                        "(filename filter bypasses distance threshold).",
                        min(k, len(pull_docs)),
                    )
                    results = {
                        "documents": [pull_docs[:k]],
                        "metadatas": [pull_metas[:k]],
                    }
                else:
                    disk_text, loaded = _load_named_sources_from_disk(
                        source_dir, source_filter
                    )
                    if disk_text:
                        logger.info(
                            "RAG loaded from disk (not in index yet): %s",
                            ", ".join(loaded),
                        )
                        return disk_text, "ok", loaded
                    logger.warning(
                        "RAG: named file(s) missing from index and disk: %s",
                        ", ".join(source_filter),
                    )
                    return "", "named_file_missing", list(source_filter)
            elif len(docs_f) < k:
                extra = _query_collection(
                    collection,
                    query_text,
                    k * 2,
                    source_filter=source_filter,
                )
                results = _merge_query_results(results, extra, k, dist_limit)
        else:
            results = _filter_results_by_distance(
                _query_collection(collection, query_text, k),
                dist_limit,
            )
    except Exception as exc:
        msg = str(exc)
        if "dimension" in msg.lower():
            logger.info(
                "RAG query skipped: index embedding dimension mismatch (re-index required)."
            )
            disk_hit = _try_named_disk_context(
                source_dir,
                base_dir,
                source_filter,
                reason="embedding dimension mismatch",
            )
            if disk_hit:
                return disk_hit
            return "", "busy", []
        logger.warning("RAG query failed: %s", exc)
        _note_rag_failure(msg)
        return "", "error", []
    finally:
        lock.release()

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
        if source_filter:
            disk_text, loaded = _load_named_sources_from_disk(
                source_dir, source_filter
            )
            if disk_text:
                logger.info(
                    "RAG loaded from disk (not in index yet): %s",
                    ", ".join(loaded),
                )
                return disk_text, "ok", loaded
            return "", "named_file_missing", list(source_filter)
        return "", "below_threshold", []

    meta_list = list(metas or [])
    while len(meta_list) < len(docs):
        meta_list.append({})

    if not source_filter:
        min_hits = open_query_min_term_hits()
        docs, meta_list = _filter_chunks_by_question_terms(
            question, docs, meta_list, min_hits=min_hits
        )
        if not docs:
            logger.info(
                "RAG: open query — no chunk matched >=%d keyword(s); "
                "no context attached (distinct terms: %s).",
                min_hits,
                ", ".join(sorted(_distinct_question_terms(question))) or "none",
            )
            return "", "below_threshold", []

    formatted = format_retrieved_passages(docs, meta_list)
    sources_used = list(dict.fromkeys(
        m.get("source", "") for m in meta_list if m and m.get("source")
    ))

    # Companion solutions: when a lecture PDF was retrieved, also load its
    # worked-solutions xlsx from disk so the model sees the full method.
    companions = find_companion_solutions(sources_used, known_sources)
    if companions:
        companion_text, companion_loaded = _load_named_sources_from_disk(
            source_dir, companions, question=question
        )
        if companion_text:
            # Wrap with a clear header so the model uses the method, not the numbers
            companion_header = (
                "[WORKED EXAMPLE — DIFFERENT SCENARIO]\n"
                "The following solution is from a different problem in the same topic.\n"
                "Use its formula structure, constants, and method for the current question.\n"
                "Do NOT copy its numerical results — recalculate using the data in the question above.\n"
            )
            companion_block = companion_header + companion_text
            cap = _prompt_max_chars()
            combined = companion_block + "\n\n" + formatted
            if len(combined) > cap:
                combined = combined[:cap].rstrip() + "\n\n[... context truncated ...]"
            formatted = combined
            sources_used = companion_loaded + [
                s for s in sources_used if s not in companion_loaded
            ]
            logger.info(
                "RAG: appended companion solution(s) from disk: %s",
                ", ".join(companion_loaded),
            )

    logger.info(
        "RAG retrieved %d chunk(s) in %.2fs from [%s] (%d chars, index %d chunks).",
        len(docs),
        elapsed,
        ", ".join(sources_used) or "unknown",
        len(formatted),
        count,
    )
    _note_rag_success()
    return formatted, "ok", sources_used


def warmup(base_dir: Path, source_dir: Path) -> None:
    """Background: rebuild stale index if needed (does not block hotkeys)."""
    done = _get_warmup_done()
    done.set()
    if not is_enabled() or not source_dir.is_dir():
        return
    try:
        consolidate_pending_index(base_dir)
        if index_is_stale(source_dir, base_dir):
            logger.info("RAG: building index in background (hotkeys stay responsive).")
        count = ensure_index(source_dir, base_dir)
        if count is not None:
            logger.info("RAG startup index built: %d chunk(s).", count)
            return

        idx_path = index_dir(base_dir)
        if not (idx_path / "chroma.sqlite3").exists():
            logger.info(
                "RAG warmup: no index yet (add files to context/)."
            )
            return

        lock = _get_lock()
        if not lock.acquire(timeout=_lock_timeout_s()):
            logger.info("RAG warmup: index busy; skipping probe.")
            return
        try:
            client = _get_client(idx_path, base_dir)
            collection = _get_collection(client, base_dir)
            n = collection.count()
        finally:
            lock.release()
        logger.info("RAG warmup complete (%d chunks indexed).", n)
    except Exception as exc:
        logger.warning("RAG warmup failed: %s", exc)


def build_context_block(question: str, source_dir: Path, base_dir: Path) -> str:
    """Retrieved context for the user prompt."""
    if not is_enabled():
        return ""
    retrieved, _status, _sources = retrieve(question, source_dir, base_dir)
    if not retrieved:
        return ""
    return retrieved
