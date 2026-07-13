"""Tests for ingest/table_classifier.py. The Anthropic client is mocked at
the boundary per AGENTS.md: no live LLM calls in the default suite."""

from __future__ import annotations

import app.ingest.table_classifier as table_classifier
from app.ingest.table_classifier import TableCategory, classify_table, render_table_as_text
from tests.conftest import make_mock_anthropic_response


def test_render_table_as_text_reconstructs_grid(sample_income_stmt_table):
    text = render_table_as_text(sample_income_stmt_table)
    lines = text.split("\n")
    assert lines[0] == "Metric | 2025 | 2024"
    assert lines[1] == "Total revenues | 94,827 | 97,690"
    assert lines[2] == "Net income | 3,855 | 7,153"


def test_classify_table_happy_path(monkeypatch, sample_income_stmt_table):
    monkeypatch.setattr(
        table_classifier.client.messages, "create",
        lambda **kw: make_mock_anthropic_response(
            '{"category": "income_stmt", "rationale": "Revenue and net income line items."}'
        ),
    )
    result = classify_table(sample_income_stmt_table)
    assert result.category == TableCategory.INCOME_STMT
    assert result.table_id == sample_income_stmt_table.table_id


def test_classify_table_fails_closed_on_garbage_response(monkeypatch, sample_income_stmt_table):
    monkeypatch.setattr(
        table_classifier.client.messages, "create",
        lambda **kw: make_mock_anthropic_response("not json at all"),
    )
    result = classify_table(sample_income_stmt_table)
    assert result.category == TableCategory.OTHER
    assert "classification_failed" in result.rationale


def test_classify_table_fails_closed_on_invalid_category(monkeypatch, sample_income_stmt_table):
    monkeypatch.setattr(
        table_classifier.client.messages, "create",
        lambda **kw: make_mock_anthropic_response(
            '{"category": "not_a_real_category", "rationale": "x"}'
        ),
    )
    result = classify_table(sample_income_stmt_table)
    assert result.category == TableCategory.OTHER
