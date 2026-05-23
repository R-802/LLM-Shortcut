"""Meta-router: classify prompts, rank providers, failover across free tiers."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm
import yaml

from router import classifier, health, quality
from router.images import ImageAttachment

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.drop_params = True

_CONFIG_PATH = Path(__file__).with_name("config.yaml")
_TIER_ORDER = ("fast", "balanced", "reasoning")


@dataclass
class RouteMeta:
    tier: str
    provider_id: str
    litellm_model: str
    attempts: int
    latency_ms: float
    escalated: bool = False


def _load_config() -> dict[str, Any]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tier_index(tier: str) -> int:
    return _TIER_ORDER.index(tier)


def _provider_api_key(env_key: str) -> str | None:
    key = os.environ.get(env_key, "").strip()
    return key or None


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "rate limit" in msg or "429" in msg:
        return True
    if "resourceexhausted" in name or "quota" in msg:
        return True
    return False


def _is_retryable_error(exc: Exception) -> bool:
    if _is_rate_limit_error(exc):
        return True
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in type(exc).__name__.lower():
        return True
    if any(code in msg for code in ("503", "502", "500", "unavailable")):
        return True
    return False


def _extract_answer(response: Any) -> str:
    try:
        choice = response.choices[0]
        content = choice.message.content
        if content:
            return str(content).strip()
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def _build_messages(
    system: str,
    user_prompt: str,
    images: list[ImageAttachment] | None = None,
) -> list[dict[str, Any]]:
    """Build messages with instructions in both system and user roles."""
    system = system.strip()
    user_body = (
        "=== INSTRUCTIONS (follow for your entire reply) ===\n"
        f"{system}\n"
        "=== END INSTRUCTIONS ===\n\n"
        f"{user_prompt.strip()}\n\n"
        "Reply now. Obey every instruction above. "
        "Maths: use +, -, *, /, and x^n only; spell pi/rho in words — no pi, rho, or multiply symbols."
    )
    if images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": user_body}]
        for img in images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": img.data_url},
                }
            )
            parts.append(
                {
                    "type": "text",
                    "text": f"(Image: {img.label})",
                }
            )
        user_content: str | list[dict[str, Any]] = parts
    else:
        user_content = user_body

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _call_provider(provider: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    env_key = provider["env_key"]
    api_key = _provider_api_key(env_key)
    if not api_key:
        raise RuntimeError(f"Missing API key for {provider['id']} ({env_key})")

    kwargs: dict[str, Any] = {
        "model": provider["litellm_model"],
        "messages": messages,
        "timeout": provider.get("timeout_s", 30),
        "max_tokens": provider.get("max_tokens", 2048),
        "temperature": provider.get("temperature", 0.3),
        "num_retries": 0,
        "api_key": api_key,
    }

    if provider["litellm_model"].startswith("openrouter/"):
        kwargs["api_base"] = "https://openrouter.ai/api/v1"

    response = litellm.completion(**kwargs)
    return _extract_answer(response)


def _build_queue(
    config: dict[str, Any],
    start_tier: str,
    *,
    vision_only: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    catalogue = config.get("providers", {})
    start_idx = _tier_index(start_tier)
    result: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    for tier in _TIER_ORDER[start_idx:]:
        tier_providers = catalogue.get(tier, [])
        if vision_only:
            tier_providers = [p for p in tier_providers if p.get("supports_vision")]
        ranked = health.rank_providers(
            tier_providers,
            provider_tier=tier,
            requested_tier=start_tier,
        )
        for p in ranked:
            if p["id"] not in seen:
                seen.add(p["id"])
                result.append((tier, p))
    return result


def complete(
    system: str,
    user_prompt: str,
    images: list[ImageAttachment] | None = None,
) -> tuple[str, RouteMeta]:
    """Route prompt to best available provider with failover."""
    health.load_state()
    config = _load_config()
    health.configure(int(config.get("cooldown_seconds", 300)))
    max_attempts = int(config.get("max_attempts", 6))

    has_images = bool(images)
    start_tier = classifier.classify(user_prompt, has_images=has_images)
    min_tier_idx = _tier_index(start_tier)
    quality_escalated = False

    messages = _build_messages(system, user_prompt, images=images)

    question_line = user_prompt
    if "Question:\n" in user_prompt:
        question_line = user_prompt.split("Question:\n", 1)[1].split("\n\nContext", 1)[0]

    queue = _build_queue(config, start_tier, vision_only=has_images)
    if not queue:
        if has_images:
            raise RuntimeError(
                "No vision-capable providers available. Add GEMINI_API_KEY or "
                "OPENROUTER_API_KEY and ensure config.yaml has supports_vision routes."
            )
        raise RuntimeError(
            "No LLM providers configured. Set API keys in .env "
            "(GEMINI_API_KEY, GROQ_API_KEY, HF_TOKEN, OPENROUTER_API_KEY)."
        )

    attempts = 0
    last_error: Exception | None = None

    for tier, provider in queue:
        tier_idx = _tier_index(tier)
        if tier_idx < min_tier_idx:
            continue
        if attempts >= max_attempts:
            break

        attempts += 1
        pid = provider["id"]
        t0 = time.perf_counter()
        try:
            answer = _call_provider(provider, messages)
            latency_ms = (time.perf_counter() - t0) * 1000

            if not answer:
                health.record_failure(pid)
                logger.warning(
                    "provider=%s tier=%s outcome=empty latency_ms=%.0f",
                    pid,
                    tier,
                    latency_ms,
                )
                last_error = RuntimeError(f"{pid} returned empty response")
                continue

            health.record_success(pid)
            logger.info(
                "provider=%s tier=%s model=%s outcome=ok latency_ms=%.0f",
                pid,
                tier,
                provider["litellm_model"],
                latency_ms,
            )

            meta = RouteMeta(
                tier=tier,
                provider_id=pid,
                litellm_model=provider["litellm_model"],
                attempts=attempts,
                latency_ms=latency_ms,
                escalated=quality_escalated or tier != start_tier,
            )

            if quality.is_weak(answer, question_line):
                nxt = classifier.next_tier(tier)
                if nxt and not quality_escalated:
                    logger.info(
                        "Weak answer from %s; escalating to tier %s",
                        pid,
                        nxt,
                    )
                    quality_escalated = True
                    min_tier_idx = _tier_index(nxt)
                    meta.escalated = True
                    continue
                return answer, meta

            return answer, meta

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            rate_limited = _is_rate_limit_error(exc)
            if _is_retryable_error(exc):
                health.record_failure(pid, rate_limited=rate_limited)
            logger.warning(
                "provider=%s tier=%s outcome=error latency_ms=%.0f err=%s",
                pid,
                tier,
                latency_ms,
                exc,
            )
            last_error = exc

    msg = "All LLM providers exhausted."
    if last_error:
        msg = f"{msg} Last error: {last_error}"
    raise RuntimeError(msg)
