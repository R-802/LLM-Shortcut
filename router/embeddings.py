"""Embedding providers for RAG: Chroma Cloud (paid) with local ONNX fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

CHROMA_EMBED_URL = "https://embed.trychroma.com/embed"
DEFAULT_CHROMA_MODEL = "BAAI/bge-m3"
DEFAULT_FALLBACK_MODELS = (
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
)
DEFAULT_INSTRUCTIONS = (
    "Represent the text for retrieval and semantic search over study materials."
)
_EMBED_BATCH = 64


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    return default


def chroma_api_key() -> str:
    return os.environ.get("CHROMA_API_KEY", "").strip()


def chroma_cloud_enabled() -> bool:
    return bool(chroma_api_key()) and _env_bool("RAG_CHROMA_EMBEDDINGS", default=True)


def chroma_embed_models() -> list[str]:
    raw = os.environ.get("RAG_CHROMA_EMBED_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    primary = os.environ.get("RAG_CHROMA_EMBED_MODEL", DEFAULT_CHROMA_MODEL).strip()
    if primary:
        return [primary, *[m for m in DEFAULT_FALLBACK_MODELS if m != primary]]
    return list(DEFAULT_FALLBACK_MODELS)


def embed_instructions() -> str:
    return os.environ.get("RAG_EMBED_INSTRUCTIONS", DEFAULT_INSTRUCTIONS).strip()


def embedding_fingerprint() -> str:
    """Stored in rag_index/manifest.json; mismatch triggers a full re-index."""
    if chroma_cloud_enabled():
        return "chroma:" + ",".join(chroma_embed_models())
    return "local:all-MiniLM-L6-v2"


def _local_onnx_ef(base_dir: Path):
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    cache_root = (
        base_dir / ".chroma_cache" / "onnx_models" / "all-MiniLM-L6-v2"
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    ef = ONNXMiniLM_L6_V2()
    ef.DOWNLOAD_PATH = cache_root
    return ef


def chroma_cloud_embed(
    texts: list[str],
    api_key: str,
    model: str,
    instructions: str,
) -> list[list[float]]:
    if not texts:
        return []
    headers = {
        "x-chroma-token": api_key,
        "x-chroma-embedding-model": model,
        "Content-Type": "application/json",
    }
    timeout = max(15.0, float(os.environ.get("RAG_EMBED_TIMEOUT_S", "60")))
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        resp = requests.post(
            CHROMA_EMBED_URL,
            headers=headers,
            json={"texts": batch, "instructions": instructions},
            timeout=timeout,
        )
        if resp.status_code == 401:
            raise RuntimeError("Chroma Cloud authentication failed (check CHROMA_API_KEY)")
        if resp.status_code == 429:
            raise RuntimeError("Chroma Cloud rate limit exceeded")
        if not resp.ok:
            detail = resp.text[:300]
            raise RuntimeError(
                f"Chroma Cloud embed HTTP {resp.status_code}: {detail}"
            )
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise RuntimeError("Chroma Cloud returned an invalid embeddings payload")
        out.extend(embeddings)
    return out


class FallbackEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Try Chroma Cloud models in order (BAAI/bge-m3 by default), then local ONNX MiniLM.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._local = _local_onnx_ef(base_dir)
        self._models = chroma_embed_models()
        self._instructions = embed_instructions()
        self._active = embedding_fingerprint()

    def __call__(self, input: Documents) -> Embeddings:
        documents = list(input)
        if not documents:
            return []

        api_key = chroma_api_key()
        if api_key and _env_bool("RAG_CHROMA_EMBEDDINGS", default=True):
            for model in self._models:
                try:
                    vectors = chroma_cloud_embed(
                        documents, api_key, model, self._instructions
                    )
                    provider = f"chroma:{model}"
                    if self._active != provider:
                        logger.info("RAG embeddings: Chroma Cloud (%s)", model)
                        self._active = provider
                    return vectors
                except Exception as exc:
                    logger.warning(
                        "Chroma Cloud embeddings failed for %s: %s",
                        model,
                        exc,
                    )

        vectors = self._local(documents)
        if self._active != "local:all-MiniLM-L6-v2":
            logger.info(
                "RAG embeddings: local ONNX all-MiniLM-L6-v2 (Chroma Cloud unavailable)"
            )
            self._active = "local:all-MiniLM-L6-v2"
        return vectors

    @staticmethod
    def name() -> str:
        return "clip_assist_fallback"

    def get_config(self) -> dict[str, Any]:
        return {
            "base_dir": str(self._base_dir.resolve()),
            "models": self._models,
            "instructions": self._instructions,
            "fingerprint": embedding_fingerprint(),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "FallbackEmbeddingFunction":
        return FallbackEmbeddingFunction(Path(config.get("base_dir", ".")))


def get_embedding_function(base_dir: Path) -> EmbeddingFunction[Documents]:
    return FallbackEmbeddingFunction(base_dir)
