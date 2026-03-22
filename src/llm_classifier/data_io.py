"""Data loading and result persistence helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .logging_utils import log
from .identity import RowKey, build_row_key, normalize_cell
from .validation import is_retryable

_EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}  # openpyxl-supported formats only


def detect_file_format(path: Path) -> str:
    """Return 'xlsx' for Excel files, 'csv' for everything else."""
    return "xlsx" if path.suffix.lower() in _EXCEL_EXTENSIONS else "csv"


def resolve_output_format(input_path: Path | None, config_format: str) -> str:
    """Determine the actual output format.

    Rules:
    - "csv" / "xlsx" → use as-is (user override)
    - "auto" → match input file format; default to "csv" if no input file
    """
    if config_format in ("csv", "xlsx"):
        return config_format
    # auto: follow input
    if input_path is not None:
        return detect_file_format(input_path)
    return "csv"


def read_input_file(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file, auto-detecting format by extension."""
    fmt = detect_file_format(path)
    if fmt == "xlsx":
        log.info(f"Reading Excel file: {path}")
        return pd.read_excel(path)
    else:
        log.info(f"Reading CSV file: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")


def write_dataframe(df: pd.DataFrame, path: Path, fmt: str = "csv") -> Path:
    """Write a DataFrame to CSV or Excel.

    Args:
        df: DataFrame to write.
        path: Output file path (extension will be adjusted to match *fmt*).
        fmt: 'csv' or 'xlsx'.

    Returns:
        The actual Path written to (may differ from *path* if extension was adjusted).
    """
    expected_ext = ".xlsx" if fmt == "xlsx" else ".csv"
    if path.suffix.lower() != expected_ext:
        path = path.with_suffix(expected_ext)

    if fmt == "xlsx":
        df.to_excel(path, index=False)
        log.info(f"Written Excel: {path} ({len(df)} rows)")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"Written CSV: {path} ({len(df)} rows)")
    return path


def deduplicate_input_rows(
    df: pd.DataFrame,
    text_column: str,
    context_column: str = "",
) -> pd.DataFrame:
    """Deduplicate rows using the same identity as run/resume/retry logic."""
    deduped = df.copy()
    text_keys = deduped[text_column].map(normalize_cell)
    if context_column:
        context_keys = deduped[context_column].map(normalize_cell)
    else:
        context_keys = pd.Series([""] * len(deduped), index=deduped.index)
    deduped["__identity_key__"] = list(zip(text_keys, context_keys))
    return (
        deduped
        .drop_duplicates(subset=["__identity_key__"])
        .drop(columns=["__identity_key__"])
        .reset_index(drop=True)
    )


def sample_programs(
    unique_df: pd.DataFrame,
    test_n: int | None = None,
    random_n: int | None = None,
) -> pd.DataFrame:
    """Sample rows from the deduplicated input DataFrame.

    Args:
        unique_df: Full DataFrame of unique items to classify.
        test_n: If set, take the first *test_n* rows (deterministic).
        random_n: If set, randomly sample *random_n* rows.

    Returns:
        Sampled (or full) DataFrame.  When neither *test_n* nor *random_n* is
        provided, the full DataFrame is returned unchanged.
    """
    if test_n is not None:
        result = unique_df.head(test_n)
        log.info(f"Test mode: taking first {test_n} rows")
    elif random_n is not None:
        n = min(random_n, len(unique_df))
        result = unique_df.sample(n=n, random_state=None).reset_index(drop=True)
        log.info(f"Random mode: sampled {n} rows")
    else:
        result = unique_df
        log.info(f"Full mode: {len(result)} rows")
    return result


def load_existing_results(
    result_path: Path,
    text_column: str,
    context_column: str = "",
) -> dict[RowKey, dict]:
    """Load existing classification results from *result_path*.

    Returns a ``{row_identity: row_dict}`` mapping. Retryable rows
    (API failures, unmatched labels) are excluded so they are re-classified
    on the next run.

    Args:
        result_path: Path to the CSV results file.
        text_column: Name of the primary text column.
        context_column: Optional secondary context column used in row identity.
    """
    if not result_path.exists():
        return {}
    existing: dict[RowKey, dict] = {}
    skipped = 0
    with open(result_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_retryable(row):
                skipped += 1
                continue
            existing[build_row_key(row, text_column, context_column)] = row
    if skipped:
        log.info(
            f"Loaded existing results: {len(existing)} successful, "
            f"{skipped} failed row(s) will be re-processed"
        )
    else:
        log.info(f"Loaded existing results: {len(existing)} row(s)")
    return existing


def load_and_split_results(
    result_path: Path, text_column: str
) -> tuple[list[dict], list[dict]]:
    """Load *result_path* and split into (successes, failures).

    Useful for ``--retry-from`` workflows where the caller needs to separate
    previously successful rows from those that need re-processing.

    Args:
        result_path: Path to the CSV results file.
        text_column: Name of the primary-key column (used only for logging).

    Returns:
        A ``(successes, failures)`` tuple of row-dict lists.
    """
    if not result_path.exists():
        return [], []
    successes: list[dict] = []
    failures: list[dict] = []
    with open(result_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_retryable(row):
                failures.append(row)
            else:
                successes.append(row)
    log.info(
        f"Split results: {len(successes)} successful, {len(failures)} failed"
    )
    return successes, failures
