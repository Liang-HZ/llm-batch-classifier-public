"""LLM API call and response parsing."""

from __future__ import annotations

import asyncio
import json
import random as random_module
from dataclasses import dataclass

import aiohttp
import json_repair

from .logging_utils import log
from .config import ClassifyConfig
from .validation import Validator
from .rate_limiter import SlidingWindowRateLimiter, CycleRateLimiter, CycleReservation


# ---------------------------------------------------------------------------
# RunStats
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    total_retries: int = 0
    total_429s: int = 0
    total_timeouts: int = 0
    total_other_errors: int = 0
    total_backoff_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    max_retries_program: str = ""
    max_retries_count: int = 0


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_api_response(raw_text: str, threshold: int, validator: Validator) -> dict:
    """Parse API JSON response, validate labels, and apply confidence filtering."""
    try:
        data = json_repair.loads(raw_text)
    except Exception:
        return {
            "label": "parse_error",
            "confidence": 0,
            "confidence_detail": raw_text,
            "is_low_confidence": "yes",
            "parse_status": "json_parse_failed",
        }

    labels = []
    if isinstance(data, dict):
        if "error" in data and not data.get("labels"):
            error_message = str(data.get("error", "unknown error"))
            return {
                "label": "api_error",
                "confidence": 0,
                "confidence_detail": json.dumps(data, ensure_ascii=False),
                "is_low_confidence": "yes",
                "parse_status": f"api_error: {error_message}",
            }
        labels = data.get("labels", [])
    elif isinstance(data, list):
        labels = data
    else:
        return {
            "label": "parse_error",
            "confidence": 0,
            "confidence_detail": raw_text,
            "is_low_confidence": "yes",
            "parse_status": f"unexpected_type: {type(data).__name__}",
        }

    if not labels:
        return {
            "label": "unclassified",
            "confidence": 0,
            "confidence_detail": json.dumps(data, ensure_ascii=False),
            "is_low_confidence": "yes",
            "parse_status": "no_labels",
        }

    validated_labels = []
    corrections = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        try:
            confidence = int(float(item.get("confidence", 0)))
        except (ValueError, TypeError):
            confidence = 0
        reason = str(item.get("reason", ""))
        if not name.strip():
            continue
        validated_name, exact = validator.validate_label(name)
        if not exact and validated_name != name:
            corrections.append(f"{name} → {validated_name}")
        validated_labels.append({
            "name": validated_name,
            "confidence": confidence,
            "reason": reason,
            "exact_match": exact,
            "original_name": name if not exact else None,
        })

    if not validated_labels:
        return {
            "label": "unclassified",
            "confidence": 0,
            "confidence_detail": json.dumps(data, ensure_ascii=False),
            "is_low_confidence": "yes",
            "parse_status": "no_valid_labels_after_validation",
        }

    validated_labels.sort(key=lambda x: x["confidence"], reverse=True)
    accepted = [lbl for lbl in validated_labels if lbl["confidence"] >= threshold]
    max_confidence = validated_labels[0]["confidence"]

    if not accepted:
        accepted = [validated_labels[0]]
        is_low = "yes"
    else:
        is_low = "no"

    final_names = []
    seen_names: set[str] = set()
    for lbl in accepted:
        if lbl["name"] in validator.category_set:
            final_name = lbl["name"]
        else:
            final_name = f"unmatched_{lbl['name']}"
        if final_name in seen_names:
            continue
        seen_names.add(final_name)
        final_names.append(final_name)

    parse_status_parts = []
    if corrections:
        parse_status_parts.append(f"fuzzy_corrected: {'; '.join(corrections)}")
    non_enum = [n for n in final_names if n.startswith("unmatched_")]
    if non_enum:
        parse_status_parts.append(f"unmatched: {'; '.join(non_enum)}")

    return {
        "label": "|".join(final_names),
        "confidence": max_confidence,
        "confidence_detail": json.dumps(validated_labels, ensure_ascii=False),
        "is_low_confidence": is_low,
        "parse_status": "; ".join(parse_status_parts) if parse_status_parts else "ok",
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


async def call_llm(
    session: aiohttp.ClientSession,
    config: ClassifyConfig,
    text: str,
    context: str,
    semaphore: asyncio.Semaphore,
    window_limiter: SlidingWindowRateLimiter | None = None,
    cycle_limiter: CycleRateLimiter | None = None,
    run_stats: RunStats | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> tuple[str, bool, dict | None]:
    """Call the LLM API. Returns (response_text, success, usage_dict|None)."""
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": config.user_prompt_template.format(
                text=text, context=context)},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    attempt = 0
    throttle_count = 0
    this_retries = 0

    while attempt < config.max_retries:
        if shutdown_event and shutdown_event.is_set():
            return '{"error": "cancelled (graceful shutdown)"}', False, None

        cycle_reservation: CycleReservation | None = None
        backoff_wait: float | None = None

        try:
            # BUG #1 FIX: cycle_limiter first (may block for next cycle),
            # then window_limiter (close to actual HTTP call, not consumed early).
            if cycle_limiter:
                cycle_reservation = await cycle_limiter.acquire(shutdown_event=shutdown_event)
                if cycle_reservation is None and shutdown_event and shutdown_event.is_set():
                    return '{"error": "cancelled (graceful shutdown)"}', False, None
            if window_limiter:
                await window_limiter.acquire(shutdown_event=shutdown_event)

            async with semaphore:
                async with session.post(
                    f"{config.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        try:
                            content = result["choices"][0]["message"]["content"].strip()
                            usage = result.get("usage")
                            if cycle_reservation:
                                await cycle_reservation.success()
                                cycle_reservation = None
                            throttle_count = 0  # BUG #2 FIX: reset on success
                            if run_stats and usage:
                                run_stats.total_prompt_tokens += usage.get("prompt_tokens", 0)
                                run_stats.total_completion_tokens += usage.get("completion_tokens", 0)
                                run_stats.total_tokens += usage.get("total_tokens", 0)
                            if run_stats and this_retries > run_stats.max_retries_count:
                                run_stats.max_retries_program = text
                                run_stats.max_retries_count = this_retries
                            return content, True, usage
                        except (KeyError, IndexError, TypeError):
                            if cycle_reservation:
                                await cycle_reservation.release()
                                cycle_reservation = None
                            throttle_count = 0  # BUG #2 FIX: reset on non-429
                            if run_stats:
                                run_stats.total_other_errors += 1
                            log.warning(f"Malformed API response structure: {str(result)[:200]}")
                            return '{"error": "malformed_api_response"}', False, None
                    elif resp.status == 429:
                        if cycle_reservation:
                            await cycle_reservation.release()
                            cycle_reservation = None
                        throttle_count += 1
                        this_retries += 1
                        if run_stats:
                            run_stats.total_429s += 1
                        if throttle_count > config.throttle_max_attempts:
                            log.error(
                                f"429 rate limit persisted for {throttle_count - 1} consecutive attempts, "
                                f"giving up: {text}"
                            )
                            return '{"error": "rate_limit_persistent"}', False, None
                        base_wait = min(
                            config.throttle_base_wait * throttle_count,
                            config.throttle_max_wait,
                        )
                        jitter = base_wait * (0.5 + random_module.random())
                        if run_stats:
                            run_stats.total_backoff_seconds += jitter
                        await resp.text()
                        log.warning(
                            f"429 rate limited (attempt {throttle_count}/{config.throttle_max_attempts}), "
                            f"waiting {jitter:.2f}s: {text}"
                        )
                        backoff_wait = jitter
                    else:
                        body = await resp.text()
                        log.warning(f"API error {resp.status}: {body[:200]}")
                        throttle_count = 0  # BUG #2 FIX: reset on non-429
                        if run_stats:
                            run_stats.total_other_errors += 1

            if backoff_wait is not None:
                if shutdown_event and shutdown_event.is_set():
                    return '{"error": "cancelled (graceful shutdown)"}', False, None
                await asyncio.sleep(backoff_wait)
                continue

            if cycle_reservation:
                await cycle_reservation.release()
                cycle_reservation = None
        except asyncio.TimeoutError:
            if cycle_reservation:
                await cycle_reservation.release()
                cycle_reservation = None
            throttle_count = 0  # BUG #2 FIX: reset on non-429
            this_retries += 1
            if run_stats:
                run_stats.total_timeouts += 1
            log.warning(f"Timeout ({attempt + 1}/{config.max_retries}): {text}")
        except asyncio.CancelledError:
            # CancelledError does not inherit from Exception; must be caught separately
            # to ensure cycle reservation is released.
            if cycle_reservation:
                await cycle_reservation.release()
                cycle_reservation = None
            raise
        except Exception as e:
            if cycle_reservation:
                await cycle_reservation.release()
                cycle_reservation = None
            throttle_count = 0  # BUG #2 FIX: reset on non-429
            this_retries += 1
            if run_stats:
                run_stats.total_other_errors += 1
            log.warning(f"Exception {e} ({attempt + 1}/{config.max_retries}): {text}")

        attempt += 1
        if run_stats:
            run_stats.total_retries += 1
        if attempt < config.max_retries:
            wait = 2 ** (attempt + 1)
            await asyncio.sleep(wait)

    if run_stats and this_retries > run_stats.max_retries_count:
        run_stats.max_retries_program = text
        run_stats.max_retries_count = this_retries
    return f'{{"error": "api_call_failed_after_{config.max_retries}_retries"}}', False, None
