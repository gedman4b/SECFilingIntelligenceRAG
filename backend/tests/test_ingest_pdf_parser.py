"""Tests for ingest/pdf_parser.py's pure structural-detection helpers.

pdf_parser.py is the reference implementation (AGENTS.md); these tests
document and guard its existing behavior rather than validate a change.
"""

from __future__ import annotations

import pytest

from app.ingest.pdf_parser import (
    _cell_is_numeric,
    _clean_cell,
    _detect_header_row,
    _looks_like_financial_table,
    parse_filing,
)


@pytest.mark.parametrize("value,expected", [
    ("1,234", True),
    ("1,234.56", True),
    ("(1,234)", True),
    ("$1,234", True),
    ("-1,234", True),
    ("12.5%", True),
    ("1", True),
    ("(0.02)", True),
    ("Total revenues", False),
    ("", False),
    ("N/A", False),
])
def test_cell_is_numeric(value, expected):
    assert _cell_is_numeric(value) == expected


def test_clean_cell_normalizes_whitespace():
    assert _clean_cell("  Total   revenues  \n") == "Total revenues"


def test_clean_cell_returns_none_for_empty():
    assert _clean_cell(None) is None
    assert _clean_cell("   ") is None


def test_looks_like_financial_table_accepts_numeric_heavy_table():
    rows = [
        ["Metric", "2025", "2024"],
        ["Total revenues", "94,827", "97,690"],
        ["Net income", "3,855", "7,153"],
    ]
    assert _looks_like_financial_table(rows) is True


def test_looks_like_financial_table_rejects_mostly_text_table():
    rows = [
        ["Item", "Description"],
        ["1", "Business overview and strategy"],
        ["1A", "Risk factors affecting the company"],
    ]
    assert _looks_like_financial_table(rows) is False


def test_looks_like_financial_table_rejects_single_row():
    assert _looks_like_financial_table([["one", "row"]]) is False


def test_detect_header_row_finds_low_numeric_density_row():
    rows = [
        ["Metric", "2025", "2024"],
        ["Total revenues", "94,827", "97,690"],
    ]
    assert _detect_header_row(rows) == 0


def test_parse_filing_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_filing("/no/such/file.pdf", "TEST-FILING")
