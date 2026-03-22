"""Helpers for building stable row identities across run/resume/retry flows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

RowKey: TypeAlias = tuple[str, str]


def normalize_cell(value: object) -> str:
    """Normalise a cell value for identity comparisons."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value)


def build_item_key(
    text_value: object,
    context_value: object = "",
    context_column: str = "",
) -> RowKey:
    """Build a stable identity key from text and optional context."""
    context_key = normalize_cell(context_value) if context_column else ""
    return normalize_cell(text_value), context_key


def build_row_key(
    row: Mapping[str, object],
    text_column: str,
    context_column: str = "",
) -> RowKey:
    """Build a stable identity key from a row-like mapping."""
    context_value = row.get(context_column, "") if context_column else ""
    return build_item_key(row.get(text_column, ""), context_value, context_column)
