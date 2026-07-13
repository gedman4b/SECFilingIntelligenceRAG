"""Tests for ingest/fact_extractor.py. The Anthropic client is mocked at
the boundary per AGENTS.md: no live LLM calls in the default suite."""

from __future__ import annotations

import pytest

import app.ingest.fact_extractor as fact_extractor
from app.ingest.fact_extractor import FilingContext, _RawFactRow, _period_guidance, _validate_row, extract_facts
from app.ingest.table_classifier import TableCategory, TableClassification
from tests.conftest import make_mock_anthropic_response


# =============================================================================
# _validate_row: pass-two rule-based validation
# =============================================================================

def test_validate_row_accepts_well_formed_row():
    row = _validate_row("t1", {
        "row_id": 0, "metric_raw_label": "Total revenues", "year": 2025,
        "value": 94827.0, "units": "USD_millions",
    })
    assert row is not None
    assert row.value == 94827.0


def test_validate_row_rejects_non_numeric_value():
    assert _validate_row("t1", {
        "row_id": 0, "metric_raw_label": "Total revenues", "year": 2025,
        "value": "not_a_number", "units": "USD_millions",
    }) is None


def test_validate_row_rejects_implausible_year():
    assert _validate_row("t1", {
        "row_id": 0, "metric_raw_label": "Total revenues", "year": 1500,
        "value": 100.0, "units": "USD_millions",
    }) is None


def test_validate_row_rejects_out_of_range_quarter():
    assert _validate_row("t1", {
        "row_id": 0, "metric_raw_label": "Total revenues", "year": 2025,
        "quarter": 5, "value": 100.0, "units": "USD_millions",
    }) is None


def test_raw_fact_row_rejects_ytd_without_quarter():
    with pytest.raises(Exception):
        _RawFactRow(
            row_id=0, metric_raw_label="Total revenues", year=2025,
            quarter=None, is_ytd=True, value=100.0, units="USD_millions",
        )


def test_raw_fact_row_accepts_ytd_with_quarter():
    row = _RawFactRow(
        row_id=0, metric_raw_label="Total revenues", year=2025,
        quarter=2, is_ytd=True, value=219659.0, units="USD_millions",
    )
    assert row.is_ytd is True
    assert row.quarter == 2


# =============================================================================
# _period_guidance: filing-context disambiguation text
# =============================================================================

def test_period_guidance_10q_explains_ytd_vs_discrete():
    text = _period_guidance(FilingContext(form_type="10-Q", fiscal_year=2026, fiscal_quarter=1))
    assert "quarter 1" in text
    assert "year-to-date" in text
    assert "is_ytd" in text


def test_period_guidance_10k_describes_full_year():
    text = _period_guidance(FilingContext(form_type="10-K", fiscal_year=2025, fiscal_quarter=None))
    assert "full fiscal year 2025" in text


# =============================================================================
# extract_facts: LLM call mocked at the boundary
# =============================================================================

def test_extract_facts_skips_other_category_without_llm_call(monkeypatch, sample_income_stmt_table):
    called = []
    monkeypatch.setattr(
        fact_extractor.client.messages, "create",
        lambda **kw: called.append(1) or make_mock_anthropic_response("{}"),
    )
    result = extract_facts(
        sample_income_stmt_table,
        TableClassification(table_id=sample_income_stmt_table.table_id, category=TableCategory.OTHER),
        FilingContext(form_type="10-K", fiscal_year=2025, fiscal_quarter=None),
    )
    assert result == []
    assert called == []


def test_extract_facts_happy_path(monkeypatch, sample_income_stmt_table):
    mock_json = '''{"facts": [
        {"row_id": 1, "metric_raw_label": "Total revenues", "year": 2025, "value": 94827, "units": "USD_millions"},
        {"row_id": 1, "metric_raw_label": "Total revenues", "year": 2024, "value": 97690, "units": "USD_millions"}
    ]}'''
    monkeypatch.setattr(
        fact_extractor.client.messages, "create",
        lambda **kw: make_mock_anthropic_response(mock_json),
    )
    facts = extract_facts(
        sample_income_stmt_table,
        TableClassification(table_id=sample_income_stmt_table.table_id, category=TableCategory.INCOME_STMT),
        FilingContext(form_type="10-K", fiscal_year=2025, fiscal_quarter=None),
    )
    assert len(facts) == 2
    assert {f.value for f in facts} == {94827.0, 97690.0}
    assert all(f.table_id == sample_income_stmt_table.table_id for f in facts)


def test_extract_facts_fails_closed_on_malformed_json(monkeypatch, sample_income_stmt_table):
    monkeypatch.setattr(
        fact_extractor.client.messages, "create",
        lambda **kw: make_mock_anthropic_response("not json"),
    )
    facts = extract_facts(
        sample_income_stmt_table,
        TableClassification(table_id=sample_income_stmt_table.table_id, category=TableCategory.INCOME_STMT),
        FilingContext(form_type="10-K", fiscal_year=2025, fiscal_quarter=None),
    )
    assert facts == []


def test_extract_facts_treats_truncated_response_as_empty(monkeypatch, sample_income_stmt_table):
    response = make_mock_anthropic_response('{"facts": [')
    response.stop_reason = "max_tokens"
    monkeypatch.setattr(fact_extractor.client.messages, "create", lambda **kw: response)
    facts = extract_facts(
        sample_income_stmt_table,
        TableClassification(table_id=sample_income_stmt_table.table_id, category=TableCategory.INCOME_STMT),
        FilingContext(form_type="10-K", fiscal_year=2025, fiscal_quarter=None),
    )
    assert facts == []
