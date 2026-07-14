"""Tests for agents/answer_composer.py. The Anthropic client is mocked at
the boundary per AGENTS.md: no live LLM calls in the default suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import app.agents.answer_composer as answer_composer
from app.schemas import Fact, Period, QueryPlan, QuestionType, Warning
from tests.conftest import make_mock_anthropic_response


def _fact(**overrides) -> Fact:
    defaults = dict(
        value=94827.0, units="USD_millions", period=Period(year=2025),
        metric_canonical_id="METRIC_TOTAL_REVENUE", metric_raw_label="Total revenues",
        is_gaap=True, filing_id="X", filing_url="u", page_number=61,
        table_id="X::p61::t1", row_id=1,
    )
    defaults.update(overrides)
    return Fact(**defaults)


def test_insufficient_data_never_calls_the_llm(monkeypatch):
    """The Composer must not spend a call (or risk inventing content) once
    the Verifier has already decided the answer is insufficient_data."""
    mock_create = MagicMock()
    monkeypatch.setattr(answer_composer.client.messages, "create", mock_create)

    plan = QueryPlan(question_type=QuestionType.NUMERIC_LOOKUP)
    response = answer_composer.compose_answer(
        question="q", plan=plan, facts=[], computed=None, comp_expr=None, prose=[],
        warnings=[Warning(severity="error", message="No matching facts found.")],
        confidence="insufficient_data",
    )
    assert mock_create.call_count == 0
    assert "cannot answer" in response.answer_text.lower()
    assert response.confidence == "insufficient_data"
    assert response.citations == []


def test_compose_answer_includes_facts_in_prompt(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured["user_msg"] = kwargs["messages"][0]["content"]
        return make_mock_anthropic_response("Tesla's total revenue was $94,827 million.")

    monkeypatch.setattr(answer_composer.client.messages, "create", fake_create)

    plan = QueryPlan(question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA")
    response = answer_composer.compose_answer(
        question="What was Tesla's revenue?", plan=plan, facts=[_fact()],
        computed=None, comp_expr=None, prose=[], warnings=[], confidence="high",
    )
    assert "94,827" in captured["user_msg"]
    assert response.citations[0]["page"] == 61
    assert response.confidence == "high"


def test_compose_answer_includes_prose_in_prompt(monkeypatch):
    from app.store.vector_store import ProsePassage

    captured = {}

    def fake_create(**kwargs):
        captured["user_msg"] = kwargs["messages"][0]["content"]
        return make_mock_anthropic_response("Tesla cites supply chain risk.")

    monkeypatch.setattr(answer_composer.client.messages, "create", fake_create)

    plan = QueryPlan(question_type=QuestionType.NARRATIVE, company_ticker="TSLA")
    passages = [ProsePassage(
        chunk_id="c1", filing_id="X", section_type="risk_factors",
        page_start=18, page_end=54, text="Battery supplier concentration risk.",
        distance=0.2,
    )]
    answer_composer.compose_answer(
        question="What risks does Tesla face?", plan=plan, facts=[],
        computed=None, comp_expr=None, prose=passages, warnings=[], confidence="high",
    )
    assert "Battery supplier concentration risk" in captured["user_msg"]


def test_compose_system_forbids_derived_arithmetic():
    """Regression guard for a real bug found in live testing: the Composer
    once silently subtracted two given facts to state an unauthorized
    dollar delta. The prompt must explicitly forbid this."""
    assert "never perform arithmetic" in answer_composer.COMPOSE_SYSTEM.lower()


def test_compose_system_forbids_confidence_mismatch():
    """Regression guard for a real bug found in live testing: given
    confidence='medium', the Composer's own narrative stated
    "Confidence: Insufficient data" -- a self-contradiction the structured
    confidence field never showed. The prompt must forbid restating
    confidence in different words than the value it was given."""
    system_lower = answer_composer.COMPOSE_SYSTEM.lower()
    assert "insufficient data" in system_lower
    assert "given value" in system_lower


def test_compose_system_forbids_estimating_missing_metrics():
    """Regression guard for a related, more severe bug found in the same
    live session: asked for a margin the system couldn't fully compute
    (revenue was never retrieved), the Composer invented an approximate
    percentage using a number it admitted was not provided. The prompt
    must forbid estimating a missing input, not just forbid combining
    inputs that were actually given."""
    system_lower = answer_composer.COMPOSE_SYSTEM.lower()
    assert "not retrieved" in system_lower or "was not retrieved" in system_lower
    assert "estimate" in system_lower
