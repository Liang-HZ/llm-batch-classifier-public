"""Tests for llm_classifier.validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_classifier.config import ClassifyConfig
from llm_classifier.validation import (
    RESULT_FIELDNAMES,
    SUCCESS_VALUES,
    Validator,
    categorize_failure,
    is_failed_result,
    is_retryable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Computer Science",
    "Finance",
    "Marketing",
    "Data Science",
    "Electrical Engineering",
]


def _make_validator() -> Validator:
    return Validator(CATEGORIES, set(CATEGORIES))


def _make_config() -> ClassifyConfig:
    return ClassifyConfig(categories=CATEGORIES)


# ---------------------------------------------------------------------------
# 1. Validator.validate_label — exact match
# ---------------------------------------------------------------------------


class TestValidatorValidateLabelExactMatch:
    def test_exact_match_returns_true(self) -> None:
        v = _make_validator()
        result, is_exact = v.validate_label("Finance")
        assert result == "Finance"
        assert is_exact is True

    def test_exact_match_multi_word(self) -> None:
        v = _make_validator()
        result, is_exact = v.validate_label("Computer Science")
        assert result == "Computer Science"
        assert is_exact is True

    def test_stripped_whitespace_still_exact(self) -> None:
        """Leading/trailing whitespace is stripped; still counts as exact."""
        v = _make_validator()
        result, is_exact = v.validate_label("  Marketing  ")
        assert result == "Marketing"
        assert is_exact is True

    def test_unknown_no_close_match_returns_original_false(self) -> None:
        v = _make_validator()
        result, is_exact = v.validate_label("Alchemy")
        assert result == "Alchemy"
        assert is_exact is False


# ---------------------------------------------------------------------------
# 2. Validator.validate_label — fuzzy match
# ---------------------------------------------------------------------------


class TestValidatorValidateLabelFuzzyMatch:
    def test_typo_corrected_fuzzy(self) -> None:
        v = _make_validator()
        # "Finace" is close enough to "Finance" at cutoff 0.7
        result, is_exact = v.validate_label("Finace")
        assert result == "Finance"
        assert is_exact is False

    def test_fuzzy_returns_false_for_is_exact(self) -> None:
        v = _make_validator()
        _, is_exact = v.validate_label("Finace")
        assert is_exact is False

    def test_very_different_name_no_fuzzy_match(self) -> None:
        v = _make_validator()
        result, is_exact = v.validate_label("Zoology")
        # Too different — no match
        assert result == "Zoology"
        assert is_exact is False

    def test_fuzzy_match_data_science_variant(self) -> None:
        v = _make_validator()
        result, is_exact = v.validate_label("Data Scienc")
        assert result == "Data Science"
        assert is_exact is False


# ---------------------------------------------------------------------------
# 3. Validator.extract_valid
# ---------------------------------------------------------------------------


class TestValidatorExtractValid:
    def test_single_valid(self) -> None:
        v = _make_validator()
        assert v.extract_valid("Finance") == ["Finance"]

    def test_pipe_separated_all_valid(self) -> None:
        v = _make_validator()
        result = v.extract_valid("Finance|Marketing")
        assert result == ["Finance", "Marketing"]

    def test_invalid_names_skipped(self) -> None:
        v = _make_validator()
        result = v.extract_valid("Finance|Alchemy|Marketing")
        assert result == ["Finance", "Marketing"]

    def test_duplicates_deduplicated(self) -> None:
        v = _make_validator()
        result = v.extract_valid("Finance|Finance|Marketing")
        assert result == ["Finance", "Marketing"]

    def test_empty_segments_skipped(self) -> None:
        v = _make_validator()
        result = v.extract_valid("Finance||Marketing")
        assert result == ["Finance", "Marketing"]

    def test_empty_string_returns_empty(self) -> None:
        v = _make_validator()
        assert v.extract_valid("") == []

    def test_all_invalid_returns_empty(self) -> None:
        v = _make_validator()
        assert v.extract_valid("Alchemy|Necromancy") == []

    def test_whitespace_trimmed_in_segments(self) -> None:
        v = _make_validator()
        result = v.extract_valid(" Finance | Marketing ")
        assert result == ["Finance", "Marketing"]


# ---------------------------------------------------------------------------
# 4. Validator.post_validate — removes invalid label names
# ---------------------------------------------------------------------------


class TestValidatorPostValidate:
    def test_valid_labels_unchanged(self, tmp_path: Path) -> None:
        v = _make_validator()
        results = [{"label": "Finance", "parse_status": "ok", "confidence_detail": ""}]
        v.post_validate(results, tmp_path)
        assert results[0]["label"] == "Finance"

    def test_invalid_names_removed_keeps_good(self, tmp_path: Path) -> None:
        v = _make_validator()
        results = [
            {
                "label": "Finance|Alchemy",
                "parse_status": "ok",
                "confidence_detail": "",
                "text": "x",
                "context": "y",
            }
        ]
        v.post_validate(results, tmp_path)
        assert results[0]["label"] == "Finance"
        assert "post_validate removed" in results[0]["parse_status"]
        assert "Alchemy" in results[0]["parse_status"]

    def test_all_invalid_becomes_unclassified(self, tmp_path: Path) -> None:
        v = _make_validator()
        results = [
            {
                "label": "Alchemy|Necromancy",
                "parse_status": "ok",
                "confidence_detail": "",
                "text": "x",
                "context": "y",
            }
        ]
        v.post_validate(results, tmp_path)
        assert results[0]["label"] == "unclassified"
        assert results[0]["is_low_confidence"] == "yes"

    def test_failure_rows_skipped(self, tmp_path: Path) -> None:
        v = _make_validator()
        results = [{"label": "unclassified", "parse_status": "api_error", "confidence_detail": ""}]
        v.post_validate(results, tmp_path)
        # Should not be mutated
        assert results[0]["label"] == "unclassified"

    def test_csv_written_when_invalids_found(self, tmp_path: Path) -> None:
        v = _make_validator()
        results = [
            {
                "label": "Finance|BadLabel",
                "parse_status": "ok",
                "confidence_detail": "",
                "text": "t",
                "context": "c",
            }
        ]
        v.post_validate(results, tmp_path)
        assert (tmp_path / "post_validation.csv").exists()

    def test_no_csv_when_all_valid(self, tmp_path: Path) -> None:
        v = _make_validator()
        results = [{"label": "Finance", "parse_status": "ok", "confidence_detail": ""}]
        v.post_validate(results, tmp_path)
        assert not (tmp_path / "post_validation.csv").exists()

    def test_from_config_factory(self, tmp_path: Path) -> None:
        cfg = _make_config()
        v = Validator.from_config(cfg)
        results = [{"label": "Finance", "parse_status": "ok", "confidence_detail": ""}]
        v.post_validate(results, tmp_path)
        assert results[0]["label"] == "Finance"

    def test_audit_csv_uses_configured_column_names(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig(
            categories=CATEGORIES,
            text_column="program_name",
            context_column="program_name_zh",
        )
        v = Validator.from_config(cfg)
        results = [
            {
                "program_name": "Finance Program",
                "program_name_zh": "金融项目",
                "label": "Finance|BadLabel",
                "parse_status": "ok",
                "confidence_detail": "",
            }
        ]
        v.post_validate(results, tmp_path)
        audit_csv = tmp_path / "post_validation.csv"
        assert audit_csv.exists()
        header = audit_csv.read_text(encoding="utf-8-sig").splitlines()[0]
        assert "program_name" in header
        assert "program_name_zh" in header


# ---------------------------------------------------------------------------
# 5. is_failed_result
# ---------------------------------------------------------------------------


class TestIsFailedResult:
    def test_ok_status_is_not_failed(self) -> None:
        assert is_failed_result({"label": "Finance", "parse_status": "ok"}) is False

    def test_api_error_is_failed(self) -> None:
        assert is_failed_result({"label": "api_error", "parse_status": "api_error"}) is True

    def test_processing_error_is_failed(self) -> None:
        assert is_failed_result({"label": "processing_error", "parse_status": "processing_error"}) is True

    def test_parse_error_is_failed(self) -> None:
        assert is_failed_result({"label": "parse_error", "parse_status": "parse_error"}) is True

    def test_unclassified_is_failed(self) -> None:
        assert is_failed_result({"label": "unclassified", "parse_status": "unclassified"}) is True

    def test_missing_parse_status_is_failed(self) -> None:
        # Missing key → empty string → not in SUCCESS_VALUES
        assert is_failed_result({}) is True

    def test_fuzzy_corrected_valid_label_is_not_failed(self) -> None:
        row = {"label": "Finance", "parse_status": "fuzzy_corrected: Finace → Finance"}
        assert is_failed_result(row) is False

    def test_ok_is_the_only_success_value(self) -> None:
        assert SUCCESS_VALUES == {"ok"}


# ---------------------------------------------------------------------------
# 6. is_retryable
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_failed_result_is_retryable(self) -> None:
        assert is_retryable({"parse_status": "api_error", "label": ""}) is True

    def test_ok_with_clean_label_not_retryable(self) -> None:
        assert is_retryable({"parse_status": "ok", "label": "Finance"}) is False

    def test_fuzzy_corrected_valid_label_not_retryable(self) -> None:
        row = {"parse_status": "fuzzy_corrected: Finace → Finance", "label": "Finance"}
        assert is_retryable(row) is False

    def test_ok_with_unmatched_label_is_retryable(self) -> None:
        row = {"parse_status": "ok", "label": "unmatched_SomeThing"}
        assert is_retryable(row) is True

    def test_ok_with_mixed_unmatched_is_retryable(self) -> None:
        row = {"parse_status": "ok", "label": "Finance|unmatched_Other"}
        assert is_retryable(row) is True

    def test_ok_with_pipe_separated_valid_labels_not_retryable(self) -> None:
        row = {"parse_status": "ok", "label": "Finance|Marketing"}
        assert is_retryable(row) is False

    def test_missing_label_with_failed_status(self) -> None:
        assert is_retryable({"parse_status": "parse_error"}) is True


# ---------------------------------------------------------------------------
# 7. categorize_failure — transient vs semantic
# ---------------------------------------------------------------------------


class TestCategorizFailureTransientVsSemantic:
    def test_api_error_label_is_transient(self) -> None:
        assert categorize_failure({"label": "api_error", "confidence_detail": ""}) == "transient"

    def test_processing_error_label_is_transient(self) -> None:
        assert categorize_failure({"label": "processing_error", "confidence_detail": ""}) == "transient"

    def test_api_error_in_detail_is_transient(self) -> None:
        assert (
            categorize_failure({"label": "unclassified", "confidence_detail": "api_error: timeout"})
            == "transient"
        )

    def test_parse_error_is_semantic(self) -> None:
        assert categorize_failure({"label": "parse_error", "confidence_detail": ""}) == "semantic"

    def test_unclassified_is_semantic(self) -> None:
        assert categorize_failure({"label": "unclassified", "confidence_detail": ""}) == "semantic"

    def test_unmatched_prefix_is_semantic(self) -> None:
        assert (
            categorize_failure({"label": "unmatched_Finance", "confidence_detail": ""}) == "semantic"
        )

    def test_empty_row_is_semantic(self) -> None:
        assert categorize_failure({}) == "semantic"


# ---------------------------------------------------------------------------
# 8. RESULT_FIELDNAMES are English
# ---------------------------------------------------------------------------


class TestResultFieldnamesAreEnglish:
    def test_contains_required_english_fields(self) -> None:
        expected = {"label", "confidence", "confidence_detail", "is_low_confidence",
                    "parse_status", "tokens_used"}
        assert expected.issubset(set(RESULT_FIELDNAMES))

    def test_no_chinese_characters(self) -> None:
        joined = "".join(RESULT_FIELDNAMES)
        for ch in joined:
            assert ord(ch) < 0x4E00 or ord(ch) > 0x9FFF, (
                f"Chinese character found in RESULT_FIELDNAMES: {ch!r}"
            )

    def test_fieldnames_order(self) -> None:
        assert RESULT_FIELDNAMES[0] == "label"
        assert RESULT_FIELDNAMES[-1] == "tokens_used"

    def test_is_low_confidence_uses_english_values(self) -> None:
        """Spot-check that post_validate writes 'yes'/'no', not '是'/'否'."""
        v = _make_validator()
        results = [
            {
                "label": "Alchemy",
                "parse_status": "ok",
                "confidence_detail": "",
                "text": "x",
                "context": "y",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            v.post_validate(results, Path(tmp))
        assert results[0].get("is_low_confidence") == "yes"
