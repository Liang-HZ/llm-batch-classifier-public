"""Label validation and post-processing for classification results.

Notes:
    post_validate() has side-effects: it mutates result dicts in-place,
    writes a CSV audit file, and logs warnings.
"""

from __future__ import annotations

import csv
import difflib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_classifier.config import ClassifyConfig

from llm_classifier.logging_utils import log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_VALUES: set[str] = {"unclassified", "parse_error", "processing_error", "api_error"}
SUCCESS_VALUES: set[str] = {"ok"}

# Result fieldnames (text_column and context_column are prepended dynamically).
RESULT_FIELDNAMES: list[str] = [
    "label",
    "confidence",
    "confidence_detail",
    "is_low_confidence",
    "parse_status",
    "tokens_used",
]

# Transient failures are API-layer errors that typically recover on retry.
_TRANSIENT_VALUES: set[str] = {"api_error", "processing_error"}


# ---------------------------------------------------------------------------
# Validator — config-driven
# ---------------------------------------------------------------------------


class Validator:
    """Label validation using categories from config."""

    def __init__(
        self,
        categories: list[str],
        category_set: set[str],
        text_column: str = "text",
        context_column: str = "",
    ) -> None:
        self.categories = categories
        self.category_set = category_set
        self.text_column = text_column
        self.context_column = context_column

    @classmethod
    def from_config(cls, config: "ClassifyConfig") -> "Validator":
        return cls(
            config.categories,
            config.category_set,
            text_column=config.text_column,
            context_column=config.context_column,
        )

    def validate_label(self, name: str) -> tuple[str, bool]:
        """Validate label name against categories.

        Returns:
            (corrected_name, is_exact_match) — exact match first, then
            stripped-whitespace match, then difflib fuzzy match (cutoff 0.7).
            If no match is found, returns (original_name, False).
        """
        if name in self.category_set:
            return name, True
        name_stripped = name.strip()
        if name_stripped in self.category_set:
            return name_stripped, True
        matches = difflib.get_close_matches(name, self.categories, n=1, cutoff=0.7)
        if matches:
            return matches[0], False
        return name, False

    def extract_valid(self, value: str) -> list[str]:
        """Extract valid category names from a pipe-separated string.

        Skips empty segments, names not in the category set, and duplicates.
        Preserves order of first occurrence.
        """
        names: list[str] = []
        seen: set[str] = set()
        for raw in str(value or "").split("|"):
            name = raw.strip()
            if not name or name not in self.category_set or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def post_validate(self, results: list[dict], run_dir: Path) -> list[dict]:
        """Post-classification hard validation against the category set.

        Mutates *results* in-place:
        - Rows with bad label names have the invalid names stripped out.
        - If all names are invalid, the label is set to ``"unclassified"``.
        - A CSV audit file is written to *run_dir*/post_validation.csv when
          invalid labels are found.

        Returns the (mutated) results list.
        """
        log.info("Post-validation: hard-matching labels against category set...")
        invalid_records: list[dict] = []

        for r in results:
            raw_labels = str(r.get("label", ""))
            if raw_labels in FAILURE_VALUES or raw_labels == "":
                continue
            names = [n.strip() for n in raw_labels.split("|")]
            bad_names = [n for n in names if n not in self.category_set]
            if bad_names:
                record = {
                    self.text_column: r.get(self.text_column, ""),
                    "raw_label": raw_labels,
                    "invalid_names": "|".join(bad_names),
                    "confidence_detail": r.get("confidence_detail", ""),
                }
                if self.context_column:
                    record[self.context_column] = r.get(self.context_column, "")
                invalid_records.append(record)
                good_names = self.extract_valid(raw_labels)
                if good_names:
                    r["label"] = "|".join(good_names)
                    r["parse_status"] = (
                        r.get("parse_status", "")
                        + f"; post_validate removed: {', '.join(bad_names)}"
                    )
                else:
                    r["label"] = "unclassified"
                    r["is_low_confidence"] = "yes"
                    r["parse_status"] = (
                        r.get("parse_status", "")
                        + f"; post_validate all_invalid: {', '.join(bad_names)}"
                    )

        validate_csv = run_dir / "post_validation.csv"
        if invalid_records:
            with open(validate_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(invalid_records[0].keys()))
                writer.writeheader()
                writer.writerows(invalid_records)
            log.warning(
                f"Post-validation: {len(invalid_records)} result(s) contained invalid "
                f"label names — see {validate_csv}"
            )
        else:
            log.info("Post-validation passed: all labels are valid category names")

        return results


# ---------------------------------------------------------------------------
# Standalone helpers (no config needed)
# ---------------------------------------------------------------------------


def is_failed_result(row: dict) -> bool:
    """Return True if *row* represents a failed classification."""
    label = str(row.get("label", "")).strip()
    if not label or label in FAILURE_VALUES:
        return True
    return any(
        segment.strip().startswith("unmatched_")
        for segment in label.split("|")
        if segment.strip()
    )


def is_retryable(row: dict) -> bool:
    """Return True if *row* should be retried.

    A row is retryable when it is a failed result.
    """
    return is_failed_result(row)


def categorize_failure(row: dict) -> str:
    """Categorise a failure row as ``'transient'`` or ``'semantic'``.

    Transient — API timeouts, 429s, processing errors — likely recoverable
    on retry.  Semantic — parse errors, unclassified, unmatched labels —
    the model could not handle the input; diminishing returns from retrying.
    """
    label = str(row.get("label", ""))
    detail = str(row.get("confidence_detail", ""))
    if label in _TRANSIENT_VALUES or "api_error" in detail:
        return "transient"
    return "semantic"
