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
from router.rag_sources import (
    chunk_body_for_prompt,
    expand_query_with_sources,
    infer_source_filters,
    prepare_chunk_for_embedding,
)
from router.stdio import patch_pythonw_stdio

logger = logging.getLogger(__name__)

_lock = None  # lazy threading.Lock() — Chroma queries + index swap only
_rebuild_schedule_lock = None  # prevents parallel full rebuilds
_collection_name = "clip_context"
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
    "about", "after", "also", "application", "applications", "approximately",
    "because", "being", "benefit", "between", "calculated", "capacity",
    "change", "compare", "conditions", "context", "conventional", "define",
    "delivered", "delivers", "describe", "difference", "discuss", "efficiency",
    "efficiencies", "energy", "engine", "engines", "everything", "explain",
    "factor", "files", "fluid", "following", "generation", "generator",
    "important", "installed", "many", "model", "much", "normally",
    "often", "operates", "operating", "other", "overall", "plant",
    "plants", "power", "question", "receive", "report", "reservoir", "results",
    "should", "show", "station", "stations", "such", "summarize", "summary",
    "system", "systems", "table", "technology", "technologies", "tell", "temperature",
    "that", "their", "there", "thermal", "these", "they", "this", "turbine",
    "under", "used", "using", "value", "what", "when", "where", "which", "while",
    "with", "would", "your",
})


def _distinct_question_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for token in re.split(r"\W+", question.lower()):
        if len(token) >= 4 and token not in _RAG_TERM_STOPWORDS and not token.isdigit():
            terms.add(token)
    return terms


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

    kept_docs: list = []
    kept_metas: list = []
    for doc, meta in zip(docs, metas):
        body = chunk_body_for_prompt(str(doc)).lower()
        if sum(1 for term in terms if term in body) >= min_hits:
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


def _populate_index_at(idx_path: Path, source_dir: Path, base_dir: Path) -> int:
    """Write chunks into a fresh index directory (does not use the live app client)."""
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

    return len(documents)


def _promote_staging_index(staging: Path, idx_path: Path, base_dir: Path) -> None:
    """Swap a finished staging index into place without deleting the live index first."""
    if idx_path.exists():
        backup = base_dir / f"rag_index_prev_{int(time.time())}"
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        try:
            shutil.move(str(idx_path), str(backup))
        except OSError:
            shutil.rmtree(idx_path, ignore_errors=True)
    shutil.move(str(staging), str(idx_path))


def _try_named_disk_context(
    source_dir: Path,
    base_dir: Path,
    source_filter: list[str],
    *,
    reason: str,
) -> tuple[str, str, list[str]] | None:
    """Load named context/ files without opening Chroma (fast path during rebuild)."""
    if not source_filter:
        return None

    manifest_files = _load_manifest(manifest_path(base_dir))
    not_indexed = [s for s in source_filter if s not in manifest_files]
    stale = index_is_stale(source_dir, base_dir)
    load_list = source_filter if stale else not_indexed
    if not load_list:
        return None

    disk_text, loaded = _load_named_sources_from_disk(source_dir, load_list)
    if not disk_text:
        return None
    logger.info("RAG loaded from disk (%s): %s", reason, ", ".join(loaded))
    return disk_text, "ok", loaded


def _load_named_sources_from_disk(
    source_dir: Path, sources: list[str]
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


def rebuild_index(source_dir: Path, base_dir: Path) -> int:
    """Re-ingest all files under source_dir. Returns chunk count."""
    source_dir = source_dir.resolve()
    base_dir = base_dir.resolve()
    idx_path = index_dir(base_dir)
    staging = base_dir / "rag_index_staging"

    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    count = _populate_index_at(staging, source_dir, base_dir)
    _save_manifest(staging / "manifest.json", _scan_signatures(source_dir))

    with _get_lock():
        _close_client()
        _promote_staging_index(staging, idx_path, base_dir)

    logger.info("RAG index built: %d chunks from %s", count, source_dir)
    return count


def index_is_stale(source_dir: Path, base_dir: Path) -> bool:
    """True when context/ files differ from the last indexed manifest."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        return False
    current = _scan_signatures(source_dir)
    previous = _load_manifest(manifest_path(base_dir))
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

    disk_hit = _try_named_disk_context(
        source_dir, base_dir, source_filter, reason="index pending or not indexed"
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
        logger.warning("RAG query failed: %s", exc)
        _note_rag_failure(str(exc))
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
