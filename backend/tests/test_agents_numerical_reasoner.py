"""Tests for agents/numerical_reasoner.py. Deterministic Python arithmetic,
never delegated to an LLM."""

from __future__ import annotations

from app.agents.numerical_reasoner import compute
from app.schemas import Fact, Period, QueryPlan, QuestionType


def _fact(year: int, value: float, quarter=None) -> Fact:
    return Fact(
        value=value, units="USD_millions", period=Period(year=year, quarter=quarter),
        metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income",
        is_gaap=True, filing_id="X", filing_url="u", page_number=1,
        table_id="X::p1::t1", row_id=1,
    )


def test_growth_calc_matches_independently_verified_value():
    """Tesla net income FY2024->FY2025, independently computed by hand:
    ((3855-7153)/|7153|)*100 = -46.106529%, rounds to -46.11."""
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, expr = compute(plan, [_fact(2024, 7153.0), _fact(2025, 3855.0)])
    assert computed == -46.11
    assert "7,153" in expr and "3,855" in expr


def test_growth_calc_positive_growth():
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, _expr = compute(plan, [_fact(2023, 96773.0), _fact(2024, 97690.0)])
    assert computed == 0.95


def test_growth_calc_undefined_when_base_is_zero():
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, message = compute(plan, [_fact(2024, 0.0), _fact(2025, 100.0)])
    assert computed is None
    assert "undefined" in message


def test_growth_calc_needs_at_least_two_facts():
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, expr = compute(plan, [_fact(2025, 100.0)])
    assert computed is None and expr is None


def test_growth_calc_sorts_by_period_regardless_of_input_order():
    """The newer period must be treated as the endpoint even if it comes
    first in the input list."""
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, _expr = compute(plan, [_fact(2025, 3855.0), _fact(2024, 7153.0)])
    assert computed == -46.11


def test_comparison_returns_raw_delta_not_percentage():
    """Independently verified: 54941-48390 = 6551."""
    plan = QueryPlan(question_type=QuestionType.COMPARISON)
    computed, expr = compute(plan, [_fact(2024, 48390.0), _fact(2025, 54941.0)])
    assert computed == 6551.0
    assert "%" not in expr


def test_comparison_delta_can_be_negative():
    plan = QueryPlan(question_type=QuestionType.COMPARISON)
    computed, _expr = compute(plan, [_fact(2024, 364980.0), _fact(2025, 359241.0)])
    assert computed == -5739.0


def test_narrative_and_numeric_lookup_never_compute():
    for qtype in (QuestionType.NARRATIVE, QuestionType.NUMERIC_LOOKUP, QuestionType.UNKNOWN):
        plan = QueryPlan(question_type=qtype)
        computed, expr = compute(plan, [_fact(2024, 100.0), _fact(2025, 200.0)])
        assert computed is None and expr is None
