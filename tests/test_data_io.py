"""Tests for llm_classifier.data_io."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from llm_classifier.data_io import (
    deduplicate_input_rows,
    detect_file_format,
    load_and_split_results,
    load_existing_results,
    read_input_file,
    resolve_output_format,
    sample_programs,
    write_dataframe,
)


TEXT_COL = "program_name"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of row dicts to a UTF-8-sig CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _result_row(
    program_name: str,
    label: str = "Finance",
    parse_status: str = "ok",
) -> dict:
    return {
        TEXT_COL: program_name,
        "label": label,
        "confidence": "99",
        "confidence_detail": "[]",
        "is_low_confidence": "no",
        "parse_status": parse_status,
        "tokens_used": "150",
    }


# ---------------------------------------------------------------------------
# load_and_split_results
# ---------------------------------------------------------------------------


def test_load_and_split_results(tmp_path: Path):
    """Successful rows go to successes; retryable rows (non-ok status) go to failures."""
    result_path = tmp_path / "results.csv"
    rows = [
        _result_row("MSc Finance", label="Finance", parse_status="ok"),
        _result_row("MSc Marketing", label="Marketing", parse_status="ok"),
        # Failed row: parse_status not in SUCCESS_VALUES
        _result_row("MSc Unknown", label="api_error", parse_status="api_error: timeout"),
        # Unmatched label row: is_retryable returns True
        _result_row("MSc X", label="unmatched_Mystery", parse_status="unmatched: unmatched_Mystery"),
    ]
    _write_csv(result_path, rows)

    successes, failures = load_and_split_results(result_path, TEXT_COL)

    assert len(successes) == 2
    assert len(failures) == 2
    success_names = {r[TEXT_COL] for r in successes}
    assert success_names == {"MSc Finance", "MSc Marketing"}
    failure_names = {r[TEXT_COL] for r in failures}
    assert failure_names == {"MSc Unknown", "MSc X"}


def test_load_and_split_results_missing_file(tmp_path: Path):
    """A missing file returns two empty lists without raising."""
    result_path = tmp_path / "nonexistent.csv"
    successes, failures = load_and_split_results(result_path, TEXT_COL)
    assert successes == []
    assert failures == []


# ---------------------------------------------------------------------------
# load_existing_results
# ---------------------------------------------------------------------------


def test_load_existing_results(tmp_path: Path):
    """Successful rows are returned; retryable rows are skipped."""
    result_path = tmp_path / "results.csv"
    rows = [
        _result_row("MSc Finance", label="Finance", parse_status="ok"),
        _result_row("MSc CS", label="Computer Science", parse_status="ok"),
        # This row is retryable (api_error parse_status)
        _result_row("MSc Bad", label="api_error", parse_status="api_error: 429"),
    ]
    _write_csv(result_path, rows)

    existing = load_existing_results(result_path, TEXT_COL)

    assert len(existing) == 2
    assert ("MSc Finance", "") in existing
    assert ("MSc CS", "") in existing
    assert ("MSc Bad", "") not in existing


def test_load_existing_results_uses_text_and_context_identity(tmp_path: Path):
    result_path = tmp_path / "results.csv"
    rows = [
        {
            TEXT_COL: "MSc Finance",
            "context": "UK campus",
            "label": "Finance",
            "confidence": "99",
            "confidence_detail": "[]",
            "is_low_confidence": "no",
            "parse_status": "ok",
            "tokens_used": "150",
        },
        {
            TEXT_COL: "MSc Finance",
            "context": "Singapore campus",
            "label": "Marketing",
            "confidence": "99",
            "confidence_detail": "[]",
            "is_low_confidence": "no",
            "parse_status": "ok",
            "tokens_used": "150",
        },
    ]
    _write_csv(result_path, rows)

    existing = load_existing_results(result_path, TEXT_COL, "context")

    assert len(existing) == 2
    assert ("MSc Finance", "UK campus") in existing
    assert ("MSc Finance", "Singapore campus") in existing


def test_load_existing_results_keeps_fuzzy_corrected_successes(tmp_path: Path):
    result_path = tmp_path / "results.csv"
    rows = [
        _result_row("MSc Finance", label="Finance", parse_status="fuzzy_corrected: Finace → Finance"),
    ]
    _write_csv(result_path, rows)

    existing = load_existing_results(result_path, TEXT_COL)

    assert "MSc Finance" in {key[0] for key in existing.keys()}


def test_load_existing_results_missing_file(tmp_path: Path):
    """A missing file returns an empty dict."""
    result_path = tmp_path / "nonexistent.csv"
    existing = load_existing_results(result_path, TEXT_COL)
    assert existing == {}


# ---------------------------------------------------------------------------
# sample_programs — test mode
# ---------------------------------------------------------------------------


def test_sample_programs_test_mode():
    """test_n returns the first N rows in original order."""
    df = pd.DataFrame({"program_name": [f"Program {i}" for i in range(20)]})
    sampled = sample_programs(df, test_n=5)
    assert len(sampled) == 5
    assert list(sampled["program_name"]) == [f"Program {i}" for i in range(5)]


def test_sample_programs_test_mode_larger_than_df():
    """test_n larger than the DataFrame returns all rows."""
    df = pd.DataFrame({"program_name": [f"Program {i}" for i in range(3)]})
    sampled = sample_programs(df, test_n=100)
    assert len(sampled) == 3


# ---------------------------------------------------------------------------
# sample_programs — random mode
# ---------------------------------------------------------------------------


def test_sample_programs_random_mode():
    """random_n returns exactly N rows chosen from the DataFrame."""
    df = pd.DataFrame({"program_name": [f"Program {i}" for i in range(50)]})
    sampled = sample_programs(df, random_n=10)
    assert len(sampled) == 10
    # All sampled values must come from the original
    assert set(sampled["program_name"]).issubset(set(df["program_name"]))


def test_sample_programs_random_mode_larger_than_df():
    """random_n larger than the DataFrame returns all rows (no error)."""
    df = pd.DataFrame({"program_name": [f"Program {i}" for i in range(5)]})
    sampled = sample_programs(df, random_n=100)
    assert len(sampled) == 5


# ---------------------------------------------------------------------------
# sample_programs — full mode
# ---------------------------------------------------------------------------


def test_sample_programs_full_mode():
    """With neither test_n nor random_n, returns the whole DataFrame."""
    df = pd.DataFrame({"program_name": [f"Program {i}" for i in range(30)]})
    sampled = sample_programs(df)
    assert len(sampled) == 30


def test_deduplicate_input_rows_uses_text_and_context():
    df = pd.DataFrame(
        [
            {"program_name": "MSc Finance", "context": "UK", "extra": 1},
            {"program_name": "MSc Finance", "context": "UK", "extra": 2},
            {"program_name": "MSc Finance", "context": "SG", "extra": 3},
        ]
    )
    deduped = deduplicate_input_rows(df, "program_name", "context")
    assert len(deduped) == 2
    assert set(zip(deduped["program_name"], deduped["context"])) == {
        ("MSc Finance", "UK"),
        ("MSc Finance", "SG"),
    }


# ---------------------------------------------------------------------------
# File format detection
# ---------------------------------------------------------------------------


def test_detect_file_format_csv():
    assert detect_file_format(Path("data.csv")) == "csv"
    assert detect_file_format(Path("data.tsv")) == "csv"
    assert detect_file_format(Path("data.txt")) == "csv"


def test_detect_file_format_excel():
    assert detect_file_format(Path("data.xlsx")) == "xlsx"
    assert detect_file_format(Path("data.XLSX")) == "xlsx"
    assert detect_file_format(Path("data.xlsm")) == "xlsx"
    # .xls and .xlsb are NOT supported (need extra engines)
    assert detect_file_format(Path("data.xls")) == "csv"
    assert detect_file_format(Path("data.xlsb")) == "csv"


def test_resolve_output_format_explicit():
    assert resolve_output_format(Path("data.csv"), "xlsx") == "xlsx"
    assert resolve_output_format(Path("data.xlsx"), "csv") == "csv"


def test_resolve_output_format_auto_follows_input():
    assert resolve_output_format(Path("data.csv"), "auto") == "csv"
    assert resolve_output_format(Path("data.xlsx"), "auto") == "xlsx"


def test_resolve_output_format_auto_no_input():
    assert resolve_output_format(None, "auto") == "csv"


# ---------------------------------------------------------------------------
# read_input_file / write_dataframe
# ---------------------------------------------------------------------------


def test_read_input_file_csv(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"text": ["A", "B"], "context": ["a", "b"]})
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    loaded = read_input_file(csv_path)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["text", "context"]


def test_read_input_file_excel(tmp_path: Path):
    xlsx_path = tmp_path / "data.xlsx"
    df = pd.DataFrame({"text": ["A", "B"], "context": ["a", "b"]})
    df.to_excel(xlsx_path, index=False)
    loaded = read_input_file(xlsx_path)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["text", "context"]


def test_write_dataframe_csv(tmp_path: Path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = tmp_path / "result.csv"
    write_dataframe(df, out, fmt="csv")
    assert out.exists()
    loaded = pd.read_csv(out, encoding="utf-8-sig")
    assert len(loaded) == 3


def test_write_dataframe_xlsx(tmp_path: Path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = tmp_path / "result.xlsx"
    write_dataframe(df, out, fmt="xlsx")
    assert out.exists()
    loaded = pd.read_excel(out)
    assert len(loaded) == 3


def test_write_dataframe_adjusts_extension(tmp_path: Path):
    """When path has .csv but format is xlsx, extension should be adjusted."""
    df = pd.DataFrame({"x": [1]})
    out = tmp_path / "result.csv"
    write_dataframe(df, out, fmt="xlsx")
    expected = tmp_path / "result.xlsx"
    assert expected.exists()
    assert not out.exists()
