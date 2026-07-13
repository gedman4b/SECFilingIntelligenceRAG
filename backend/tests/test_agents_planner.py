"""Tests for agents/planner.py. The Anthropic client is mocked at the
boundary per AGENTS.md: no live LLM calls in the default suite."""

from __future__ import annotations

import app.agents.planner as planner
from app.schemas import QuestionType
from tests.conftest import make_mock_tool_use_response


def test_plan_query_happy_path(monkeypatch):
    monkeypatch.setattr(
        planner.client.messages, "create",
        lambda **kw: make_mock_tool_use_response("submit_query_plan", {
            "question_type": "numeric_lookup",
            "company_ticker": "TSLA",
            "metric_natural_language": "total revenue",
            "periods": [{"year": 2025}],
        }),
    )
    plan = planner.plan_query("What was Tesla's total revenue in fiscal year 2025?")
    assert plan.question_type == QuestionType.NUMERIC_LOOKUP
    assert plan.company_ticker == "TSLA"
    assert plan.metric_natural_language == "total revenue"
    assert plan.periods[0].year == 2025


def test_plan_query_fails_closed_on_invalid_tool_input(monkeypatch):
    """Regression test for a real bug found in live testing: the model
    returning field names that don't match QueryPlan must not crash the
    request, it must fail closed to UNKNOWN with an ambiguity flag."""
    monkeypatch.setattr(
        planner.client.messages, "create",
        lambda **kw: make_mock_tool_use_response("submit_query_plan", {
            "question_type": "numeric_lookup",
            "periods": [{"fiscal_year": 2025}],  # wrong field name
        }),
    )
    plan = planner.plan_query("What was Tesla's total revenue in fiscal year 2025?")
    assert plan.question_type == QuestionType.UNKNOWN
    assert len(plan.ambiguity_flags) == 1
    assert "plan_parse_failed" in plan.ambiguity_flags[0]


def test_plan_query_fails_closed_on_client_exception(monkeypatch):
    def raise_error(**kw):
        raise RuntimeError("network error")
    monkeypatch.setattr(planner.client.messages, "create", raise_error)
    plan = planner.plan_query("What was Tesla's total revenue?")
    assert plan.question_type == QuestionType.UNKNOWN


def test_plan_tool_schema_matches_query_plan_fields():
    """The tool's input_schema is generated from QueryPlan itself, so this
    mostly guards against someone hand-rolling a stale schema later."""
    props = planner.PLAN_TOOL["input_schema"]["properties"]
    assert "question_type" in props
    assert "periods" in props
    assert "metric_natural_language" in props
