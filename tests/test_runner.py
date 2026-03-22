"""Tests for llm_classifier.runner — run() and run_retry()."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pytest

import llm_classifier.runner as runner_module
from llm_classifier.config import ClassifyConfig
from llm_classifier.runner import (
    run,
    run_retry,
    _count_failure_types,
    _filter_retryable,
    _update_semantic_fail_streak,
    _write_back_results,
)
from llm_classifier.validation import RESULT_FIELDNAMES


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
        "max_retries": 1,
        "throttle_max_attempts": 10,
        "throttle_base_wait": 30.0,
        "throttle_max_wait": 300.0,
        "text_column": "text",
        "context_column": "context",
        "threshold": 95,
        "concurrency": 4,
        "output_dir": "",  # will be overridden per test
    }
    defaults.update(overrides)
    return ClassifyConfig(**defaults)


def _make_result_csv(
    path: Path,
    rows: list[dict],
    text_column: str = "text",
    context_column: str = "context",
) -> None:
    """Write a results CSV with the standard fieldnames."""
    fieldnames = [text_column, context_column] + RESULT_FIELDNAMES
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _success_row(text: str, label: str = "Finance", ctx: str = "") -> dict:
    return {
        "text": text,
        "context": ctx,
        "label": label,
        "confidence": "99",
        "confidence_detail": "[]",
        "is_low_confidence": "no",
        "parse_status": "ok",
        "tokens_used": "150",
    }


def _failure_row(text: str, label: str = "unclassified", ctx: str = "") -> dict:
    return {
        "text": text,
        "context": ctx,
        "label": label,
        "confidence": "0",
        "confidence_detail": "{}",
        "is_low_confidence": "yes",
        "parse_status": "no_labels",
        "tokens_used": "",
    }


def _api_error_row(text: str, ctx: str = "") -> dict:
    return {
        "text": text,
        "context": ctx,
        "label": "api_error",
        "confidence": "0",
        "confidence_detail": "{}",
        "is_low_confidence": "yes",
        "parse_status": "api_error: timeout",
        "tokens_used": "",
    }


# ---------------------------------------------------------------------------
# test_dry_run_does_not_create_run_dir
# ---------------------------------------------------------------------------


def test_dry_run_does_not_create_run_dir(tmp_path: Path):
    """dry_run=True must log and return without creating a run_* directory."""
    import pandas as pd

    # Create a minimal input CSV
    input_csv = tmp_path / "input.csv"
    df = pd.DataFrame([{"text": f"Item {i}", "context": f"Ctx {i}"} for i in range(5)])
    df.to_csv(input_csv, index=False, encoding="utf-8-sig")

    config = _test_config(
        input_file=str(input_csv),
        output_dir=str(tmp_path / "out"),
    )

    run(config, dry_run=True)

    # No run_* subdirectory should have been created
    out_dir = tmp_path / "out"
    run_dirs = list(out_dir.glob("run_*")) if out_dir.exists() else []
    assert run_dirs == [], f"Expected no run dirs, found: {run_dirs}"


# ---------------------------------------------------------------------------
# test_retry_dry_run_no_api_call
# ---------------------------------------------------------------------------


def test_retry_dry_run_no_api_call(tmp_path: Path):
    """run_retry with dry_run=True must not create subdirs or make API calls."""
    result_csv = tmp_path / "results.csv"
    _make_result_csv(result_csv, [_failure_row("ItemA"), _failure_row("ItemB")])

    config = _test_config(output_dir=str(tmp_path))

    run_retry(config, retry_from=result_csv, max_rounds=3, dry_run=True)

    # No run_*_retry_* subdirectory should have been created
    retry_dirs = list(tmp_path.glob("run_*_retry_*"))
    assert retry_dirs == [], f"Expected no retry dirs, found: {retry_dirs}"


# ---------------------------------------------------------------------------
# test_retry_no_failures_exits_immediately
# ---------------------------------------------------------------------------


def test_retry_no_failures_exits_immediately(tmp_path: Path):
    """run_retry must return immediately when the results CSV has no failures."""
    result_csv = tmp_path / "results.csv"
    _make_result_csv(result_csv, [_success_row("ItemA"), _success_row("ItemB")])

    config = _test_config(output_dir=str(tmp_path))

    # Should not raise or create any subdirectories
    run_retry(config, retry_from=result_csv, max_rounds=3, dry_run=False)

    retry_dirs = list(tmp_path.glob("run_*_retry_*"))
    assert retry_dirs == [], f"Expected no retry dirs, found: {retry_dirs}"


# ---------------------------------------------------------------------------
# test_retry_mutual_exclusion
# ---------------------------------------------------------------------------


def test_retry_mutual_exclusion_input_csv_and_append(tmp_path: Path):
    """run() with both input_csv and append must call sys.exit(1)."""
    import pandas as pd

    input_csv = tmp_path / "input.csv"
    pd.DataFrame([{"text": "X", "context": "Y"}]).to_csv(input_csv, index=False)

    config = _test_config(output_dir=str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        run(config, input_csv=str(input_csv), append=True)

    assert exc_info.value.code == 1


def test_retry_mutual_exclusion_input_csv_and_test(tmp_path: Path):
    """run() with both input_csv and test must call sys.exit(1)."""
    import pandas as pd

    input_csv = tmp_path / "input.csv"
    pd.DataFrame([{"text": "X", "context": "Y"}]).to_csv(input_csv, index=False)

    config = _test_config(output_dir=str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        run(config, input_csv=str(input_csv), test=5)

    assert exc_info.value.code == 1


def test_run_deduplicates_by_text_and_context_before_run_core(monkeypatch, tmp_path: Path):
    import pandas as pd

    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        [
            {"text": "Same", "context": "A"},
            {"text": "Same", "context": "A"},
            {"text": "Same", "context": "B"},
            {"text": "Other", "context": "C"},
        ]
    ).to_csv(input_csv, index=False, encoding="utf-8-sig")

    captured = {}

    def fake_run_core(**kwargs):
        captured["unique_df"] = kwargs["unique_df"].copy()

    monkeypatch.setattr(runner_module, "_run_core", fake_run_core)

    config = _test_config(
        input_file=str(input_csv),
        output_dir=str(tmp_path / "out"),
    )
    run(config)

    unique_df = captured["unique_df"]
    assert len(unique_df) == 3
    assert set(zip(unique_df["text"], unique_df["context"])) == {
        ("Same", "A"),
        ("Same", "B"),
        ("Other", "C"),
    }


# ---------------------------------------------------------------------------
# test_filter_retryable_skips_exhausted_semantic
# ---------------------------------------------------------------------------


def test_filter_retryable_skips_exhausted_semantic():
    """_filter_retryable must skip items with semantic_fail_streak >= 2."""
    failures = [
        _failure_row("ItemA"),  # semantic
        _failure_row("ItemB"),  # semantic
        _api_error_row("ItemC"),  # transient
    ]

    # ItemA has 2 semantic failures — should be skipped
    # ItemC is transient — always retried regardless of streak
    semantic_fail_streak = {("ItemA", ""): 2, ("ItemC", ""): 5}

    retryable = _filter_retryable(failures, semantic_fail_streak, text_column="text")

    texts = [r["text"] for r in retryable]
    assert "ItemA" not in texts, "ItemA should be skipped (semantic streak >= 2)"
    assert "ItemB" in texts, "ItemB should be included (no streak)"
    assert "ItemC" in texts, "ItemC should be included (transient — always retried)"


def test_filter_retryable_semantic_streak_below_threshold():
    """Items with semantic streak < 2 must still be retried."""
    failures = [_failure_row("ItemA"), _failure_row("ItemB")]
    semantic_fail_streak = {("ItemA", ""): 1}  # below threshold

    retryable = _filter_retryable(failures, semantic_fail_streak, text_column="text")

    texts = [r["text"] for r in retryable]
    assert "ItemA" in texts
    assert "ItemB" in texts


def test_filter_retryable_uses_text_and_context_identity():
    failures = [
        _failure_row("ItemA", ctx="EN"),
        _failure_row("ItemA", ctx="ZH"),
    ]
    semantic_fail_streak = {("ItemA", "EN"): 2}

    retryable = _filter_retryable(
        failures,
        semantic_fail_streak,
        text_column="text",
        context_column="context",
    )

    assert {r["context"] for r in retryable} == {"ZH"}


def test_update_semantic_fail_streak_resets_on_transient_failure():
    semantic_fail_streak = {("ItemA", "CTX"): 1}
    still_failed = [_api_error_row("ItemA", ctx="CTX")]

    _update_semantic_fail_streak(
        semantic_fail_streak,
        new_successes=[],
        still_failed=still_failed,
        text_column="text",
        context_column="context",
    )

    assert semantic_fail_streak == {}


# ---------------------------------------------------------------------------
# test_count_failure_types
# ---------------------------------------------------------------------------


def test_count_failure_types_basic():
    """_count_failure_types must categorise failure labels correctly."""
    failures = [
        _failure_row("A", label="unclassified"),
        _failure_row("B", label="parse_error"),
        _failure_row("C", label="parse_error"),
        _api_error_row("D"),  # api_error
        _failure_row("E", label="processing_error"),
        _failure_row("F", label="unmatched_XYZ"),
    ]

    counts = _count_failure_types(failures)

    assert counts.get("unclassified") == 1
    assert counts.get("parse_error") == 2
    assert counts.get("api_error") == 1
    assert counts.get("processing_error") == 1
    assert counts.get("unmatched") == 1


def test_count_failure_types_empty():
    """_count_failure_types on an empty list must return an empty dict."""
    assert _count_failure_types([]) == {}


def test_count_failure_types_other():
    """Labels that don't match known categories must be grouped as 'other'."""
    failures = [_failure_row("A", label="weird_unknown_label")]
    counts = _count_failure_types(failures)
    assert counts.get("other") == 1


# ---------------------------------------------------------------------------
# test_write_back_results_atomic
# ---------------------------------------------------------------------------


def test_write_back_results_atomic(tmp_path: Path):
    """_write_back_results must write all rows atomically and clean up .tmp."""
    result_csv = tmp_path / "results.csv"

    config = _test_config(output_dir=str(tmp_path))

    successes = [_success_row("ItemA"), _success_row("ItemB")]
    failures = [_failure_row("ItemC")]

    _write_back_results(result_csv, successes, failures, config)

    assert result_csv.exists(), "Result CSV must exist after write-back"
    tmp_path_candidate = result_csv.with_suffix(".csv.tmp")
    assert not tmp_path_candidate.exists(), ".tmp file must be removed after atomic rename"

    with open(result_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    texts = [r["text"] for r in rows]
    assert "ItemA" in texts
    assert "ItemB" in texts
    assert "ItemC" in texts


def test_write_back_results_preserves_order(tmp_path: Path):
    """_write_back_results must write successes first, then failures."""
    result_csv = tmp_path / "results.csv"
    config = _test_config(output_dir=str(tmp_path))

    successes = [_success_row("S1"), _success_row("S2")]
    failures = [_failure_row("F1"), _failure_row("F2")]

    _write_back_results(result_csv, successes, failures, config)

    with open(result_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["text"] == "S1"
    assert rows[1]["text"] == "S2"
    assert rows[2]["text"] == "F1"
    assert rows[3]["text"] == "F2"


def test_run_retry_round_records_measured_duration(monkeypatch, tmp_path: Path):
    captured = {}

    async def fake_run_classification(**kwargs):
        await asyncio.sleep(0.01)
        return [_success_row("ItemA", ctx="CtxA")]

    def fake_generate_report(*args, **kwargs):
        return None

    def fake_generate_run_summary(results, run_stats, run_dir, config, duration_seconds):
        captured["duration"] = duration_seconds

    monkeypatch.setattr(runner_module, "run_classification", fake_run_classification)
    monkeypatch.setattr(runner_module, "generate_report", fake_generate_report)
    monkeypatch.setattr(runner_module, "generate_run_summary", fake_generate_run_summary)

    run_dir = tmp_path / "run_retry"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)

    config = _test_config(output_dir=str(tmp_path))
    runner_module._run_retry_round(
        config=config,
        retryable=[_failure_row("ItemA", ctx="CtxA")],
        run_dir=run_dir,
        log_dir=log_dir,
        output_dir=tmp_path,
        concurrency=1,
    )

    assert captured["duration"] > 0
