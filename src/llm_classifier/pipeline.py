"""Batch classification core pipeline."""

from __future__ import annotations

import asyncio
import csv
import random
import time
from pathlib import Path

import aiohttp
import pandas as pd

from .logging_utils import log
from .api import call_llm, parse_api_response, RunStats
from .validation import is_failed_result, Validator, RESULT_FIELDNAMES
from .rate_limiter import SlidingWindowRateLimiter, CycleRateLimiter
from .config import ClassifyConfig
from .identity import RowKey, build_item_key


async def run_classification(
    config: ClassifyConfig,
    unique_df: pd.DataFrame,
    existing: dict[RowKey, dict],
    result_path: Path,
    window_limiter: SlidingWindowRateLimiter | None = None,
    cycle_limiter: CycleRateLimiter | None = None,
    run_stats: RunStats | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> list[dict]:
    """Classify all items in unique_df via LLM API and return results.

    Args:
        config: ClassifyConfig with model, categories, and runtime settings.
        unique_df: DataFrame with one row per unique item to classify.
        existing: Dict keyed by text_column value for checkpoint resumption.
        result_path: Path to append CSV results (created if absent).
        window_limiter: Optional sliding-window rate limiter.
        cycle_limiter: Optional cycle-based rate limiter.
        run_stats: Optional RunStats for accumulating API call statistics.
        shutdown_event: Optional asyncio.Event; when set, pending tasks are
            cancelled gracefully.
    """
    text_col = config.text_column
    context_col = config.context_column
    concurrency = config.concurrency or 15

    validator = Validator.from_config(config)

    # CSV fieldnames: prepend text/context columns to the standard result fields
    if context_col:
        fieldnames = [text_col, context_col] + RESULT_FIELDNAMES
    else:
        fieldnames = [text_col] + RESULT_FIELDNAMES

    todo_rows = []
    results = []
    for _, row in unique_df.iterrows():
        text_val = row[text_col]
        context_val = row[context_col] if context_col else ""
        row_key = build_item_key(text_val, context_val, context_col)
        if row_key in existing:
            results.append(existing[row_key])
            continue
        item: dict = {text_col: text_val}
        if context_col:
            item[context_col] = context_val
        todo_rows.append(item)

    if existing:
        log.info(
            f"Checkpoint resume: skipping {len(results)} existing results, "
            f"processing {len(todo_rows)} items"
        )
    else:
        log.info(f"Items to process: {len(todo_rows)}")

    if not todo_rows:
        log.info("All items already classified — no API calls needed")
        return results

    # Append to result_path (key for checkpoint resumption: write row by row)
    file_exists = result_path.exists() and result_path.stat().st_size > 0
    csv_file = open(result_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    # Semaphore as defense-in-depth alongside worker pool concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    counter = {
        "completed": 0,
        "success": 0,
        "failed": 0,
        "cancelled": 0,
        "last_log_time": 0.0,
    }
    total = len(todo_rows)
    start_time = time.time()

    async def process_one(session: aiohttp.ClientSession, item: dict) -> dict:
        text_val = item[text_col]
        context_val = item.get(context_col, "") if context_col else ""

        if shutdown_event and shutdown_event.is_set():
            parsed = {
                "label": "processing_error",
                "confidence": 0,
                "confidence_detail": "graceful shutdown",
                "is_low_confidence": "yes",
                "parse_status": "cancelled",
                "tokens_used": "",
            }
            result_row = {text_col: text_val, **parsed}
            if context_col:
                result_row[context_col] = context_val
            async with lock:
                counter["cancelled"] += 1
                results.append(result_row)
            return result_row

        try:
            raw, api_success, usage = await call_llm(
                session,
                config,
                text_val,
                context_val,
                semaphore,
                window_limiter=window_limiter,
                cycle_limiter=cycle_limiter,
                run_stats=run_stats,
                shutdown_event=shutdown_event,
            )
            parsed = parse_api_response(raw, config.threshold, validator)
            if usage:
                parsed["tokens_used"] = usage.get("total_tokens", "")
            else:
                parsed["tokens_used"] = ""
        except Exception as e:
            log.error(f"Processing exception {e}: {text_val}")
            parsed = {
                "label": "processing_error",
                "confidence": 0,
                "confidence_detail": str(e),
                "is_low_confidence": "yes",
                "parse_status": f"exception: {e}",
                "tokens_used": "",
            }

        result_row = {text_col: text_val, **parsed}
        if context_col:
            result_row[context_col] = context_val

        async with lock:
            writer.writerow(result_row)
            csv_file.flush()
            results.append(result_row)

            counter["completed"] += 1
            if is_failed_result(parsed):
                counter["failed"] += 1
            else:
                counter["success"] += 1

            c = counter["completed"]
            now = time.time()
            since_last = now - counter["last_log_time"]
            if c == total or c % 10 == 0 or since_last >= 5:
                counter["last_log_time"] = now
                elapsed = now - start_time
                pct = c / total * 100
                speed = c / elapsed if elapsed > 0 else 0
                eta = (total - c) / speed if speed > 0 else 0
                cycle_info = ""
                if cycle_limiter and cycle_limiter.max_success > 0:
                    cycle_info = (
                        f" | cycle {cycle_limiter.cycle_number}: "
                        f"{cycle_limiter.success_count}+{cycle_limiter.reserved_count}"
                        f"/{cycle_limiter.max_success}"
                    )
                log.info(
                    f"Progress: {c}/{total} ({pct:.1f}%) | "
                    f"success: {counter['success']} | failed: {counter['failed']} | "
                    f"speed: {speed:.1f}/s | ETA: {eta:.0f}s{cycle_info}"
                )

        return result_row

    # Worker pool: create exactly `concurrency` workers that drain the queue.
    # Avoids spawning thousands of coroutines that would all hit the rate limiter.
    queue: asyncio.Queue[dict] = asyncio.Queue()
    for item in todo_rows:
        queue.put_nowait(item)

    async def worker(session: aiohttp.ClientSession) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if config.jitter_seconds > 0 and not (
                shutdown_event and shutdown_event.is_set()
            ):
                jitter = random.uniform(0, config.jitter_seconds)
                log.info(f"Jitter {jitter:.2f}s: {item[text_col]}")
                await asyncio.sleep(jitter)
            await process_one(session, item)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            num_workers = min(concurrency, len(todo_rows))
            workers = [
                asyncio.create_task(worker(session)) for _ in range(num_workers)
            ]
            await asyncio.gather(*workers)
    finally:
        csv_file.close()

    cancelled = counter["cancelled"]
    if cancelled:
        log.warning(f"Graceful shutdown: {cancelled} item(s) cancelled")

    return results
