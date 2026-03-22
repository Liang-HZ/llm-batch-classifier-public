"""Tests for llm_classifier.pipeline — run_classification."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pandas as pd

import llm_classifier.pipeline as pipeline_module
from llm_classifier.config import ClassifyConfig
from llm_classifier.pipeline import run_classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_config(**overrides) -> ClassifyConfig:
    """Return a minimal ClassifyConfig suitable for testing."""
    defaults = {
        "categories": ["Computer Science", "Finance", "Marketing"],
        "prompt_template": "Classify into:\n{categories}",
        "user_prompt_template": "{text} / {context}",
        "model_name": "test-model",
        "api_base": "https://test.api",
        "api_key": "test-key",
        "temperature": 0.1,
        "max_tokens": 500,
        "timeout": 30,
        "max_retries": 3,
        "throttle_max_attempts": 10,
        "throttle_base_wait": 30.0,
        "throttle_max_wait": 300.0,
        "text_column": "program_name",
        "context_column": "program_name_zh",
        "threshold": 95,
        "concurrency": 8,
    }
    defaults.update(overrides)
    return ClassifyConfig(**defaults)


# ---------------------------------------------------------------------------
# test_run_classification_handles_large_fake_batch
# ---------------------------------------------------------------------------


def test_run_classification_handles_large_fake_batch(monkeypatch, tmp_path: Path):
    """run_classification must process all items and write them to CSV,
    while respecting the concurrency cap."""
    unique_df = pd.DataFrame(
        [
            {"program_name": f"Program {i}", "program_name_zh": f"Program ZH {i}"}
            for i in range(120)
        ]
    )
    result_path = tmp_path / "results.csv"
    config = _test_config(concurrency=8)

    in_flight = {"current": 0, "max": 0}

    async def fake_call_llm(
        session,
        cfg,
        text,
        context,
        semaphore,
        window_limiter=None,
        cycle_limiter=None,
        run_stats=None,
        shutdown_event=None,
    ):
        async with semaphore:
            in_flight["current"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["current"])
            await asyncio.sleep(0.001)
            in_flight["current"] -= 1
        label = "Finance"
        return (
            f'{{"labels": [{{"name": "{label}", "confidence": 99, "reason": "fake"}}]}}',
            True,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

    monkeypatch.setattr(pipeline_module, "call_llm", fake_call_llm)

    async def scenario():
        return await run_classification(
            config=config,
            unique_df=unique_df,
            existing={},
            result_path=result_path,
            window_limiter=None,
            cycle_limiter=None,
        )

    results = asyncio.run(scenario())

    assert len(results) == 120
    assert in_flight["max"] <= 8

    with open(result_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120
    assert all(row["label"] == "Finance" for row in rows)


# ---------------------------------------------------------------------------
# test_graceful_shutdown_cancels_pending_tasks
# ---------------------------------------------------------------------------


def test_graceful_shutdown_cancels_pending_tasks(monkeypatch, tmp_path: Path):
    """When shutdown_event is set, process_one must skip the API call
    and mark items as cancelled."""
    unique_df = pd.DataFrame(
        [
            {"program_name": f"Program {i}", "program_name_zh": f"ZH {i}"}
            for i in range(10)
        ]
    )
    result_path = tmp_path / "results.csv"
    config = _test_config(concurrency=4)

    call_count = {"n": 0}

    async def fake_call_llm(
        session,
        cfg,
        text,
        context,
        semaphore,
        window_limiter=None,
        cycle_limiter=None,
        run_stats=None,
        shutdown_event=None,
    ):
        if shutdown_event and shutdown_event.is_set():
            return '{"error": "cancelled"}', False, None
        call_count["n"] += 1
        return (
            '{"labels": [{"name": "Finance", "confidence": 99, "reason": "ok"}]}',
            True,
            None,
        )

    monkeypatch.setattr(pipeline_module, "call_llm", fake_call_llm)

    async def scenario():
        shutdown_event = asyncio.Event()
        shutdown_event.set()  # signal shutdown immediately
        return await run_classification(
            config=config,
            unique_df=unique_df,
            existing={},
            result_path=result_path,
            shutdown_event=shutdown_event,
        )

    results = asyncio.run(scenario())

    # All 10 items must be accounted for (as cancelled rows)
    assert len(results) == 10
    # The shutdown check in process_one fires before call_llm is reached
    assert call_count["n"] == 0


# ---------------------------------------------------------------------------
# test_jitter_adds_delay_between_requests
# ---------------------------------------------------------------------------


def test_jitter_adds_delay_between_requests(monkeypatch, tmp_path: Path):
    """When jitter_seconds > 0, asyncio.sleep must be called between items."""
    unique_df = pd.DataFrame(
        [
            {"program_name": f"Program {i}", "program_name_zh": f"ZH {i}"}
            for i in range(5)
        ]
    )
    result_path = tmp_path / "results.csv"
    # Use jitter=1.0 so random.uniform(0, 1.0) is always > 0 given mock
    config = _test_config(concurrency=1, jitter_seconds=1.0)

    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await real_sleep(0)  # yield control without actually sleeping

    async def fake_call_llm(
        session,
        cfg,
        text,
        context,
        semaphore,
        window_limiter=None,
        cycle_limiter=None,
        run_stats=None,
        shutdown_event=None,
    ):
        return (
            '{"labels": [{"name": "Finance", "confidence": 99, "reason": "ok"}]}',
            True,
            None,
        )

    monkeypatch.setattr(pipeline_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(pipeline_module.asyncio, "sleep", fake_sleep)
    # Fix random.uniform to always return 0.5 so every item gets a jitter sleep
    monkeypatch.setattr(pipeline_module.random, "uniform", lambda a, b: 0.5)

    async def scenario():
        return await run_classification(
            config=config,
            unique_df=unique_df,
            existing={},
            result_path=result_path,
        )

    results = asyncio.run(scenario())

    assert len(results) == 5
    # Each of the 5 items should have triggered a jitter sleep of 0.5
    jitter_sleeps = [d for d in sleep_calls if d == 0.5]
    assert len(jitter_sleeps) == 5
