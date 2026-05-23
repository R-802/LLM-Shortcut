"""Provider health scoring, cooldowns, and optional persistence."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent / "state.json"
_lock = threading.Lock()

TIER_ORDER = ("fast", "balanced", "reasoning")


@dataclass
class ProviderRuntime:
    cooldown_until: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    hourly_requests: list[float] = field(default_factory=list)


_runtime: dict[str, ProviderRuntime] = {}
_cooldown_seconds = 300


def configure(cooldown_seconds: int) -> None:
    global _cooldown_seconds
    _cooldown_seconds = cooldown_seconds


def _get(provider_id: str) -> ProviderRuntime:
    if provider_id not in _runtime:
        _runtime[provider_id] = ProviderRuntime()
    return _runtime[provider_id]


def load_state() -> None:
    if not STATE_PATH.is_file():
        return
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        now = time.time()
        with _lock:
            for pid, data in raw.get("providers", {}).items():
                rt = _get(pid)
                rt.cooldown_until = float(data.get("cooldown_until", 0))
                if rt.cooldown_until < now:
                    rt.cooldown_until = 0.0
                rt.failure_count = int(data.get("failure_count", 0))
                rt.success_count = int(data.get("success_count", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Could not load router state: %s", exc)


def save_state() -> None:
    with _lock:
        payload = {
            "providers": {
                pid: {
                    "cooldown_until": rt.cooldown_until,
                    "failure_count": rt.failure_count,
                    "success_count": rt.success_count,
                }
                for pid, rt in _runtime.items()
            }
        }
    try:
        STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save router state: %s", exc)


def is_on_cooldown(provider_id: str) -> bool:
    with _lock:
        return _get(provider_id).cooldown_until > time.time()


def record_success(provider_id: str) -> None:
    now = time.time()
    with _lock:
        rt = _get(provider_id)
        rt.success_count += 1
        rt.failure_count = max(0, rt.failure_count - 1)
        rt.hourly_requests = [t for t in rt.hourly_requests if now - t < 3600]
        rt.hourly_requests.append(now)
    save_state()


def record_failure(provider_id: str, rate_limited: bool = False) -> None:
    now = time.time()
    with _lock:
        rt = _get(provider_id)
        rt.failure_count += 1
        if rate_limited or rt.failure_count >= 2:
            rt.cooldown_until = now + _cooldown_seconds
    save_state()


def _quota_hint(provider_id: str) -> float:
    now = time.time()
    with _lock:
        rt = _get(provider_id)
        recent = [t for t in rt.hourly_requests if now - t < 3600]
        rt.hourly_requests = recent
        count = len(recent)
    # Prefer less-used providers within the hour (0 requests -> 1.0).
    return max(0.1, 1.0 - min(count / 60.0, 0.9))


def _health_score(provider_id: str) -> float:
    with _lock:
        rt = _get(provider_id)
        if rt.cooldown_until > time.time():
            return 0.0
        penalty = min(rt.failure_count * 0.15, 0.6)
        return max(0.1, 1.0 - penalty)


def score_provider(
    provider_id: str,
    provider_tier: str,
    requested_tier: str,
) -> float:
    health = _health_score(provider_id)
    if health <= 0:
        return -1.0
    quota = _quota_hint(provider_id)
    tier_idx = {t: i for i, t in enumerate(TIER_ORDER)}
    req_i = tier_idx.get(requested_tier, 1)
    prov_i = tier_idx.get(provider_tier, 1)
    if prov_i == req_i:
        model_fit = 1.0
    elif prov_i > req_i:
        model_fit = 0.7
    else:
        model_fit = 0.5
    with _lock:
        error_penalty = min(_get(provider_id).failure_count * 0.1, 0.5)
    return 0.4 * health + 0.3 * quota + 0.3 * model_fit - error_penalty


def rank_providers(
    providers: list[dict[str, Any]],
    provider_tier: str,
    requested_tier: str,
) -> list[dict[str, Any]]:
    available = []
    for p in providers:
        pid = p["id"]
        if is_on_cooldown(pid):
            continue
        env_key = p.get("env_key", "")
        if env_key and not os.environ.get(env_key, "").strip():
            continue
        sc = score_provider(pid, provider_tier, requested_tier)
        if sc < 0:
            continue
        available.append((sc, p))
    available.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in available]
